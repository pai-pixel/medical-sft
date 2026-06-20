"""C 数据 smoke: hjw 8 卡 跑 50 条, max_tokens 拉满, 看真实 p99
钩子 SOP step 2: 没历史可信数据 → smoke 让模型说够
输出: smoke_50.jsonl + 打印 token 分布 + finish_reason 分布
"""
import os, json, time, re, resource
import multiprocessing as mp
from queue import Empty
from pathlib import Path
import random

_, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(262144, _h), _h))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

NUM_GPUS = 8
MODEL = "/mnt/data/huangjiawei/models/Baichuan-M2-32B"
A_DATA = Path("/mnt/data/huangjiawei/datasets_local/medical_v4/chosen_v4_filtered.jsonl")
OUT_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v4")
OUT = OUT_DIR / "v4_C_smoke_50.jsonl"

N_SMOKE = 50
# === max_tokens 拉满 (max_model_len 8192 - prompt+system buffer 1024) ===
MAX_MODEL_LEN = 8192
MAX_TOKENS = 7168

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


def select_evasive(target_n: int):
    out = []
    with A_DATA.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            ans = x["messages"][1]["content"]
            if is_evasive(ans):
                out.append({
                    "src_id": x["id"],
                    "prompt": x["messages"][0]["content"],
                })
    random.seed(42)
    random.shuffle(out)
    return out[:target_n]


def worker(shard_id, queue, out_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(shard_id)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.3, max_tokens=MAX_TOKENS, top_p=0.9)
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

        rendered = []
        for item in batch:
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
            out.write(json.dumps({
                "src_id": item["src_id"],
                "prompt": item["prompt"],
                "new_answer": o.outputs[0].text,
                "finish_reason": o.outputs[0].finish_reason,
                "tok_count": len(o.outputs[0].token_ids),
            }, ensure_ascii=False) + "\n")
        done += len(batch)
        print(f"[shard {shard_id}] +{len(batch)} done={done}", flush=True)
    out.close()
    print(f"[shard {shard_id}] FINISHED", flush=True)


def analyze():
    """跑完后, 算钩子 SOP 步骤 [2] 的统计"""
    from collections import Counter
    finishes = Counter()
    toks = []
    with OUT.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            finishes[x["finish_reason"]] += 1
            toks.append(x["tok_count"])
    toks.sort()
    n = len(toks)
    print(f"\n{'='*60}")
    print(f"[SMOKE 结果] {n} 条")
    print(f"  finish_reason: {dict(finishes)}")
    print(f"  output_tokens: p50={toks[n//2]} p95={toks[int(n*0.95)]} "
          f"p99={toks[int(n*0.99)]} max={toks[-1]}")
    import math
    p99 = toks[int(n*0.99)]
    suggest = math.ceil(max(p99 * 1.15, toks[-1] + 256) / 256) * 256
    print(f"  → 钩子 SOP: max_tokens 全量建议 = max(p99×1.15, smoke_max+256) = {suggest}")
    if finishes.get("length", 0) > 0:
        print(f"  ⚠ smoke 中仍有 {finishes['length']} 条撞顶 max={MAX_TOKENS}, "
              f"实际需求 > {MAX_TOKENS} tok, 考虑加大 max_model_len 或接受截断")
    print(f"{'='*60}")


def main():
    t0 = time.time()
    items = select_evasive(N_SMOKE)
    print(f"[smoke] {len(items)} prompts, max_tokens={MAX_TOKENS}", flush=True)

    if OUT.exists():
        OUT.unlink()

    mp.set_start_method("spawn", force=True)
    queue = mp.Queue(maxsize=NUM_GPUS * 4)

    procs = []
    for sid in range(NUM_GPUS):
        p = mp.Process(target=worker, args=(sid, queue, str(OUT)))
        p.start()
        procs.append(p)

    # smoke 50 条, chunk 7 让 work-stealing 触发
    CHUNK = 7
    for i in range(0, len(items), CHUNK):
        queue.put(items[i:i+CHUNK])
    for _ in range(NUM_GPUS):
        queue.put(None)

    for p in procs:
        p.join()

    print(f"[smoke] 完成 {time.time()-t0:.0f}s", flush=True)
    analyze()


if __name__ == "__main__":
    main()
