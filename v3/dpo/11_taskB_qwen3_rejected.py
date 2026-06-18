"""Task B v2: Qwen3-8B 生成 rejected, sharded 多进程版.
- 1 节点 8 GPU 跑 8 个 TP=1 vllm 实例 (各占 1 卡)
- 数据 8-way split, 各 shard 独立 staging
- 8x throughput vs TP=8 单实例
"""
import os, json, resource, shutil, time, argparse

_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(262144, _h), _h))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"
os.environ["VLLM_USE_V1"] = "0"

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--num_shards", type=int, required=True)
ap.add_argument("--tp", type=int, default=1)
args = ap.parse_args()

print(f"[task-b shard {args.shard}/{args.num_shards}] TP={args.tp}, "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = "/mnt/data/huangjiawei/models/Qwen3-8B"
INP = "/mnt/data/huangjiawei/datasets_local/medical_dpo/prompts_pool_43k.jsonl"
LOCAL_INP = "/tmp/prompts_pool_43k.jsonl"
OUT_DIR = "/mnt/data/huangjiawei/datasets_local/medical_dpo"
STAGING = os.path.join(OUT_DIR, f"rejected_qwen3.jsonl.staging.shard{args.shard}")

if not os.path.exists(LOCAL_INP):
    shutil.copy(INP, LOCAL_INP)

# 全部 prompts (43.5K), 按 id % num_shards 分流
prompts = []
with open(LOCAL_INP, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["id"] % args.num_shards == args.shard:
            prompts.append(r)
print(f"[shard {args.shard}] my prompts: {len(prompts)}", flush=True)

done = set()
if os.path.exists(STAGING):
    with open(STAGING, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line); done.add(r["id"])
            except Exception: continue
print(f"[shard {args.shard}] resume: {len(done)} done", flush=True)
todo = [p for p in prompts if p["id"] not in done]
print(f"[shard {args.shard}] todo: {len(todo)}", flush=True)

if not todo:
    print(f"[shard {args.shard}] all done", flush=True)
    raise SystemExit(0)

print(f"[shard {args.shard}] loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"[shard {args.shard}] loading vllm Qwen3-8B TP={args.tp} bf16...", flush=True)
t0 = time.time()
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=args.tp,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    dtype="bfloat16",
    enforce_eager=True,
    trust_remote_code=True,
)
print(f"[shard {args.shard}] vllm ready in {time.time()-t0:.0f}s", flush=True)

prompts_text = []
for p in todo:
    text = tok.apply_chat_template(
        [{"role": "user", "content": p["prompt"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    prompts_text.append(text)

sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096)

CHUNK = 800
fout = open(STAGING, "a", encoding="utf-8")
t1 = time.time()
for ci in range(0, len(todo), CHUNK):
    chunk_todo = todo[ci:ci+CHUNK]
    chunk_text = prompts_text[ci:ci+CHUNK]
    print(f"[shard {args.shard}] gen {ci}/{len(todo)}...", flush=True)
    outputs = llm.generate(chunk_text, sampling)
    for i, out in enumerate(outputs):
        item = chunk_todo[i]
        full = out.outputs[0].text
        if "</think>" in full:
            parts = full.split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            answer = parts[1].strip()
        else:
            thinking = ""; answer = full.strip()
        fout.write(json.dumps({
            "id": item["id"], "prompt": item["prompt"], "domain": item["domain"],
            "thinking": thinking, "rejected": answer, "student": "qwen3-8b",
        }, ensure_ascii=False) + "\n")
    fout.flush()
    el = time.time() - t1
    done_now = ci + len(chunk_todo)
    eta = el / done_now * (len(todo) - done_now) if done_now > 0 else 0
    print(f"[shard {args.shard}] done {done_now}/{len(todo)}, "
          f"el={el/60:.1f}m ETA={eta/60:.0f}m", flush=True)
fout.close()
print(f"[shard {args.shard}] FINISHED in {(time.time()-t1)/60:.1f}m", flush=True)
