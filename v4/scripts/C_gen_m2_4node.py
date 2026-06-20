"""C 数据生成 v2 - DLC 4 节点 × 8 卡 = 32 卡, 修 max_tokens 截断
节点间: prompts[RANK::WORLD_SIZE] 静态 stride 分片 (长 prompt 均匀)
节点内: 8 卡 TP=1 + work-stealing queue (无 stragglers)
输出: v4_C_chosen.rank{N}.jsonl  (4 个文件, 后续 cat 合并)
"""
import os, json, re, time, hashlib, resource
import multiprocessing as mp
from queue import Empty
from pathlib import Path
import random

_, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(262144, _h), _h))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# === DLC 多节点 env (DLC 自动注入) ===
RANK = int(os.environ.get("RANK", os.environ.get("NODE_RANK", 0)))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("NNODES", 1)))
NUM_GPUS = 8

MODEL = "/mnt/data/huangjiawei/models/Baichuan-M2-32B"
A_DATA = Path("/mnt/data/huangjiawei/datasets_local/medical_v4/chosen_v4_filtered.jsonl")
OUT_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v4")
OUT = OUT_DIR / f"v4_C_chosen.rank{RANK}.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_C = 5000
CHUNK = 100

# === 修 1: max_tokens=4608 (smoke 50 实测 p99=3828, 钩子 SOP × 1.15 后 4608, 向上 256 整) ===
MAX_TOKENS = 4608
# === 修 2: M2 thinking 关不掉, 直接接受 + 后期切除 ===

EVADE_PATTERNS = [
    r"必须\s*(?:立即|首先|强调).*?(?:就医|医生|医院|急诊|120)",
    r"建议您?\s*(?:立即|尽快)\s*(?:就医|去医院|看医生|联系医生)",
    r"请务必\s*(?:立即|尽快).*?(?:就医|医生|医院|急诊)",
    r"^[^。!]*请咨询专业医[生师]",
    r"^[^。!]*无法替代医[生师]",
    r"^[^。!]*仅供参考",
]
EVADE_RE = re.compile("|".join(EVADE_PATTERNS))

SYSTEM_C = """你是一位资深临床医师,精通中医辨证施治与现代循证医学。

下面是一位患者的咨询场景,请按以下硬规则给出回答:

1. 必须给出明确的临床判断或鉴别诊断方向(信息不足时按现有信息给最可能的判断,再说明需补什么检查)
2. 必须给出具体的方剂/药品、剂量、用法
3. 必须列出禁忌、警示、不适合人群
4. 严禁以"建议立即就医""请咨询医生""无法替代医生面诊""仅供参考"等回避语作为开头或主体
5. 危急情况(休克/急性心肌梗死/严重外伤/意识障碍等)可建议急救,但要先给出院前可执行的处理方案
6. 全中文回答,不掺英文药名缩写(可在中文药名后括号附英文)
"""


def is_evasive(answer: str) -> bool:
    return bool(EVADE_RE.search(answer[:300]))


def select_all_prompts(target_n: int):
    """从 A 找 evasive prompts, 全局一致(用 sort + seed 保证 4 节点拿到同一份)"""
    out = []
    with A_DATA.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            ans = x["messages"][1]["content"]
            if is_evasive(ans):
                out.append({
                    "src_id": x["id"],
                    "src_file": x.get("src_file", ""),
                    "domain": x.get("domain", ""),
                    "prompt": x["messages"][0]["content"],
                })
    # 排序保证 4 节点选出同一 5000
    out.sort(key=lambda r: r["src_id"])
    if len(out) > target_n:
        random.seed(42)
        random.shuffle(out)
        out = out[:target_n]
        out.sort(key=lambda r: r["src_id"])
    return out


def worker(shard_id: int, queue: mp.Queue, out_path: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(shard_id)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.3, max_tokens=MAX_TOKENS, top_p=0.9)
    print(f"[node{RANK} shard{shard_id}] vllm loaded", flush=True)

    out = open(out_path, "a", buffering=1, encoding="utf-8")
    done = 0
    while True:
        try:
            batch = queue.get(timeout=20)
        except Empty:
            break
        if batch is None:
            break

        rendered = []
        for item in batch:
            try:
                msg = tok.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_C},
                        {"role": "user", "content": item["prompt"]},
                    ],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                msg = tok.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_C},
                        {"role": "user", "content": item["prompt"]},
                    ],
                    tokenize=False, add_generation_prompt=True,
                )
            rendered.append(msg)

        outputs = llm.generate(rendered, sampling)
        # 钩子 SOP step 4: 撞顶率告警
        length_cnt = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
        pct = length_cnt / len(outputs) * 100
        for item, o in zip(batch, outputs):
            rec = {
                "src_id": item["src_id"],
                "prompt": item["prompt"],
                "new_answer": o.outputs[0].text.strip(),
                "finish_reason": o.outputs[0].finish_reason,
                "tok_count": len(o.outputs[0].token_ids),
                "node": RANK,
                "shard": shard_id,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += len(batch)
        alert = " ⚠ ALERT" if pct > 2.0 else ""
        print(f"[node{RANK} shard{shard_id}] +{len(batch)} done={done} length={length_cnt}/{len(batch)} ({pct:.1f}%){alert}", flush=True)
    out.close()
    print(f"[node{RANK} shard{shard_id}] FINISHED done={done}", flush=True)


def main():
    t_start = time.time()
    print(f"[node {RANK}/{WORLD_SIZE}] launching, NUM_GPUS={NUM_GPUS}, MAX_TOKENS={MAX_TOKENS}", flush=True)

    all_prompts = select_all_prompts(TARGET_C)
    print(f"[node {RANK}/{WORLD_SIZE}] global {len(all_prompts)} prompts", flush=True)

    # === 节点间静态 stride 分片 (长 prompt 均匀) ===
    my = all_prompts[RANK::WORLD_SIZE]
    print(f"[node {RANK}/{WORLD_SIZE}] my share: {len(my)} prompts", flush=True)

    # 清空旧 rank 文件
    if OUT.exists():
        OUT.unlink()

    # 节点内 8 卡 work-stealing queue
    mp.set_start_method("spawn", force=True)
    queue: mp.Queue = mp.Queue(maxsize=NUM_GPUS * 4)

    procs = []
    for sid in range(NUM_GPUS):
        p = mp.Process(target=worker, args=(sid, queue, str(OUT)))
        p.start()
        procs.append(p)
        print(f"[node {RANK}] launched shard {sid}", flush=True)

    # feeder
    fed = 0
    for i in range(0, len(my), CHUNK):
        batch = my[i:i+CHUNK]
        queue.put(batch)
        fed += len(batch)
    for _ in range(NUM_GPUS):
        queue.put(None)
    print(f"[node {RANK}] feeder done, queued {fed}", flush=True)

    # monitor
    while any(p.is_alive() for p in procs):
        time.sleep(60)
        n_lines = 0
        if OUT.exists():
            with OUT.open(encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
        elapsed = time.time() - t_start
        eta = elapsed / max(n_lines, 1) * (len(my) - n_lines) if n_lines > 0 else 0
        alive = sum(1 for p in procs if p.is_alive())
        print(f"[node {RANK} monitor] T+{elapsed:.0f}s output={n_lines}/{len(my)} "
              f"alive_shards={alive}/{NUM_GPUS} ETA={eta:.0f}s", flush=True)

    for p in procs:
        p.join()
    print(f"[node {RANK}] all shards done, total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
