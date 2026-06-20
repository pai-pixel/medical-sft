"""C 数据生成: 从 v4 chosen filtered 找出 evasive 答案的 prompt,
用 M2-32B in-process 重新生成 non-evasive 答案。

策略:
- 输入: chosen_v4_filtered.jsonl (37k, 来自 A 数据过滤)
- 检测 evasive 答案 → 抽 5000 prompt
- 8 卡 work-stealing queue, M2-32B TP=1
- 强 prompt 强制不回避
- chunk=100 写盘 + 进度 print + ETA

输出: v4_C_chosen.jsonl (~5k, 后续再过滤"回避型未修好")
"""
import json, os, time, re, hashlib
import multiprocessing as mp
import resource
from queue import Empty
from pathlib import Path

# fd 上限
_, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(262144, _h), _h))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 路径
A_DATA = Path("/mnt/data/huangjiawei/datasets_local/medical_v4/chosen_v4_filtered.jsonl")
OUT_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v4")
OUT = OUT_DIR / "v4_C_chosen.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "/mnt/data/huangjiawei/models/Baichuan-M2-32B"
NUM_GPUS = 8
TARGET_C = 5000
CHUNK = 100  # work-stealing queue 每个 task 包多少 prompt

# evasive 检测规则 (跟 A_01 一致)
EVADE_PATTERNS = [
    r"必须\s*(?:立即|首先|强调).*?(?:就医|医生|医院|急诊|120)",
    r"建议您?\s*(?:立即|尽快)\s*(?:就医|去医院|看医生|联系医生)",
    r"请务必\s*(?:立即|尽快).*?(?:就医|医生|医院|急诊)",
    r"^[^。!]*请咨询专业医[生师]",
    r"^[^。!]*无法替代医[生师]",
    r"^[^。!]*仅供参考",
]
EVADE_RE = re.compile("|".join(EVADE_PATTERNS))

# 强 prompt — 强制不回避
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
    head = answer[:300]
    return bool(EVADE_RE.search(head))


def select_evasive_prompts(target_n: int):
    """从 A 37k 找 evasive 的 prompt"""
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
                    "old_answer_head": ans[:200],
                })
    print(f"[A] {len(out)} evasive prompts found", flush=True)
    if len(out) > target_n:
        import random
        random.seed(42)
        out = random.sample(out, target_n)
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
    sampling = SamplingParams(temperature=0.3, max_tokens=2000, top_p=0.9)
    print(f"[shard {shard_id}] vllm loaded", flush=True)

    out = open(out_path, "a", buffering=1, encoding="utf-8")
    done = 0
    while True:
        try:
            batch = queue.get(timeout=20)
        except Empty:
            break
        if batch is None:
            break

        # 构造 prompt
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
        for item, o in zip(batch, outputs):
            ans = o.outputs[0].text.strip()
            finish = o.outputs[0].finish_reason
            tok_count = len(o.outputs[0].token_ids)
            rec = {
                "src_id": item["src_id"],
                "prompt": item["prompt"],
                "new_answer": ans,
                "finish_reason": finish,
                "tok_count": tok_count,
                "shard": shard_id,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += len(batch)
        print(f"[shard {shard_id}] +{len(batch)} done={done}", flush=True)
    out.close()
    print(f"[shard {shard_id}] FINISHED done={done}", flush=True)


def main():
    items = select_evasive_prompts(TARGET_C)
    n = len(items)

    if OUT.exists():
        OUT.unlink()
    print(f"[start] M2 8 卡 work-stealing, {n} prompts", flush=True)

    mp.set_start_method("spawn", force=True)
    queue: mp.Queue = mp.Queue(maxsize=NUM_GPUS * 4)

    procs = []
    for sid in range(NUM_GPUS):
        p = mp.Process(target=worker, args=(sid, queue, str(OUT)))
        p.start()
        procs.append(p)
        print(f"[main] launched shard {sid}", flush=True)

    # feeder
    t0 = time.time()
    fed = 0
    for i in range(0, n, CHUNK):
        batch = items[i:i+CHUNK]
        queue.put(batch)
        fed += len(batch)
        if fed % 500 == 0 or fed == n:
            elapsed = time.time() - t0
            print(f"[feeder] queued {fed}/{n} elapsed={elapsed:.0f}s", flush=True)
    for _ in range(NUM_GPUS):
        queue.put(None)
    print(f"[feeder] poison pills sent", flush=True)

    # 等所有 shard 完成 + 主进程进度监控
    while any(p.is_alive() for p in procs):
        time.sleep(60)
        n_lines = 0
        if OUT.exists():
            with OUT.open(encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
        elapsed = time.time() - t0
        eta = elapsed / max(n_lines, 1) * (n - n_lines) if n_lines > 0 else 0
        alive = sum(1 for p in procs if p.is_alive())
        print(f"[monitor] T+{elapsed:.0f}s output={n_lines}/{n} alive_shards={alive}/{NUM_GPUS} ETA={eta:.0f}s",
              flush=True)

    for p in procs:
        p.join()
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
