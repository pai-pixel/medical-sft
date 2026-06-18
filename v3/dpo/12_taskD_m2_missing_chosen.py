"""Task D v2: M2-32B 跑 missing chosen via work-stealing queue.

变更原因 (2026-06-17 第二次跑实测):
- 旧版用 id % 8 static sharding, prompts_pool_43k 缺口 id 分布严重不均
- shard 0/4 各 4564/4617 条 ≈ 其他 shard (2310-2377) 的 2 倍
- 6/8 shard 14:50 完成, shard 0/4 还要跑 2.5h
- 净浪费 ≈ 19 GPU-h (6 卡 × 3.1h 闲置)

新版结构:
- 1 主进程 + 8 worker (mp.Process), 每 worker 独占 1 GPU TP=1
- 共享 mp.Queue(maxsize=32), feeder 切 BATCH=16 灌入, worker 抢任务
- 0 stragglers, 全自动负载均衡
- 写盘 buffering=1 行缓冲 + 每 batch flush, ≈30-60s 一次写盘 (旧版 chunk=500 是 30+ min)
- staging 命名 chosen_taskd_m2.jsonl.staging.shard{0..7} 跟旧版兼容, 直接 resume 旧 staging
"""
import os
import json
import resource
import shutil
import time
from collections import Counter

# fd ulimit
_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(262144, _h), _h))

# offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"
os.environ["VLLM_USE_V1"] = "0"

MODEL_PATH = "/mnt/data/huangjiawei/models/Baichuan-M2-32B"
OUT_DIR = "/mnt/data/huangjiawei/datasets_local/medical_dpo"
INP = os.path.join(OUT_DIR, "prompts_pool_43k.jsonl")
LOCAL_INP = "/tmp/prompts_pool_43k.jsonl"
EXIST_M2 = os.path.join(OUT_DIR, "chosen_m2_18k.jsonl")
EXIST_OPUS = os.path.join(OUT_DIR, "chosen_opus_25k.jsonl")
STAGING_TMPL = os.path.join(OUT_DIR, "chosen_taskd_m2.jsonl.staging.shard{sid}")

NUM_GPUS = 8
BATCH = 16          # queue 一批多大, 16 是 sweet spot
QUEUE_MAXSIZE = NUM_GPUS * 4   # 32 batch ≈ 512 prompt 在 queue, 反压主进程


def worker(shard_id, queue, out_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(shard_id)
    from queue import Empty
    from vllm import LLM, SamplingParams

    print(f"[shard {shard_id}] loading vllm M2 TP=1 bf16...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.93,
        max_model_len=6144,
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=3072)
    print(f"[shard {shard_id}] vllm ready in {time.time()-t0:.0f}s", flush=True)

    out = open(out_path, "a", encoding="utf-8", buffering=1)  # 行缓冲
    done = 0
    t_first = time.time()
    while True:
        try:
            batch = queue.get(timeout=30)
        except Empty:
            print(f"[shard {shard_id}] queue empty timeout, exit", flush=True)
            break
        if batch is None:
            print(f"[shard {shard_id}] poison pill, exit", flush=True)
            break

        # batch = [(id, prompt, domain, formatted_text), ...]
        formatted = [b[3] for b in batch]
        try:
            outputs = llm.generate(formatted, sampling)
        except Exception as e:
            print(f"[shard {shard_id}] generate FAILED batch_size={len(batch)}: {e}",
                  flush=True)
            continue

        for i, o in enumerate(outputs):
            item = batch[i]
            full = o.outputs[0].text
            if "</think>" in full:
                parts = full.split("</think>", 1)
                thinking = parts[0].replace("<think>", "").strip()
                answer = parts[1].strip()
            else:
                thinking = ""
                answer = full.strip()
            out.write(json.dumps({
                "id": item[0],
                "prompt": item[1],
                "domain": item[2],
                "thinking": thinking,
                "chosen": answer,
                "teacher": "baichuan-m2-32b",
            }, ensure_ascii=False) + "\n")
        out.flush()
        done += len(batch)
        el = time.time() - t_first
        rate = done / el if el > 0 else 0
        print(f"[shard {shard_id}] +{len(batch)} done={done} "
              f"el={el/60:.1f}m rate={rate*60:.1f}/min", flush=True)

    out.close()
    print(f"[shard {shard_id}] FINISHED done={done}", flush=True)


def main():
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # 串行 pre-cp 已由 launcher 做, 这里仅 fallback
    if not os.path.exists(LOCAL_INP):
        shutil.copy(INP, LOCAL_INP)

    # 1) 收集已有 chosen ids (Task A M2 + Task C Opus)
    existing_ids = set()
    for f_path in (EXIST_M2, EXIST_OPUS):
        if os.path.exists(f_path):
            local_p = f"/tmp/{os.path.basename(f_path)}"
            if not os.path.exists(local_p):
                shutil.copy(f_path, local_p)
            with open(local_p, encoding="utf-8") as f:
                for line in f:
                    try:
                        existing_ids.add(json.loads(line)["id"])
                    except Exception:
                        continue
    print(f"[main] existing chosen ids (A+C): {len(existing_ids)}", flush=True)

    # 2) 收集 staging 已 done (resume), 跨 8 个 shard 取并集
    done_ids = set()
    for sid in range(NUM_GPUS):
        sf = STAGING_TMPL.format(sid=sid)
        if os.path.exists(sf):
            with open(sf, encoding="utf-8") as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        continue
    print(f"[main] staging done ids (resume): {len(done_ids)}", flush=True)

    # 3) 加载 tokenizer 应用 chat template
    print(f"[main] loading tokenizer for chat template...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    missing = []
    domain_count = Counter()
    total_pool = 0
    with open(LOCAL_INP, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total_pool += 1
            pid = r["id"]
            if pid in existing_ids or pid in done_ids:
                continue
            text = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode="on",
            )
            missing.append((pid, r["prompt"], r["domain"], text))
            domain_count[r["domain"]] += 1

    total = len(missing)
    print(f"[main] pool={total_pool}, existing(A+C)={len(existing_ids)}, "
          f"resume_done={len(done_ids)}, missing={total} "
          f"by_domain={dict(domain_count)}", flush=True)

    if total == 0:
        print(f"[main] all done, nothing to do", flush=True)
        return

    # 4) 起 queue + workers
    queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
    procs = []
    for sid in range(NUM_GPUS):
        out_path = STAGING_TMPL.format(sid=sid)
        p = mp.Process(target=worker, args=(sid, queue, out_path))
        p.start()
        procs.append(p)
    print(f"[main] spawned {NUM_GPUS} workers, will feed {total} prompts "
          f"in batches of {BATCH}", flush=True)

    # 5) feeder: 切 BATCH 灌 queue (主进程 block 在 queue.put 反压)
    t_start = time.time()
    feed_done = 0
    PRINT_EVERY = max(BATCH * 20, 320)  # 每 ~320 prompt print 一次
    next_report = PRINT_EVERY
    for i in range(0, total, BATCH):
        batch = missing[i:i + BATCH]
        queue.put(batch)
        feed_done += len(batch)
        if feed_done >= next_report or feed_done == total:
            el = time.time() - t_start
            print(f"[main] fed {feed_done}/{total} into queue, "
                  f"elapsed={el/60:.1f}m", flush=True)
            next_report = feed_done + PRINT_EVERY

    # 毒丸通知 worker 退出
    for _ in range(NUM_GPUS):
        queue.put(None)
    print(f"[main] all batches fed + poison pills sent, "
          f"waiting for workers to drain...", flush=True)

    for p in procs:
        p.join()

    el_total = time.time() - t_start
    print(f"[main] all workers exited, total elapsed={el_total/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
