"""Task E: M2-32B 重跑 4,307 条 bad chosen (Phase 2 截断的 A bad 747 + D bad 3,560).

变更原因 (2026-06-17 重跑):
- 旧 max_tokens=3072 (Task D) / 4096 (Task A) 都不够, smoke 实测 p99=5513
- 新设 max_tokens=6400, max_model_len=8192
- 按 SOP 写盘必带 finish_reason + output_tokens
- 全量跑时每 batch print 撞顶率到 stdout

拓扑: 8 worker × TP=1 + 共享 mp.Queue (work-stealing).
Resume: 读 staging 已 done ids, 跳过.
"""
import os
import json
import resource
import shutil
import time
from collections import Counter

_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(262144, _h), _h))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MODELSCOPE_OFFLINE'] = '1'
os.environ['VLLM_USE_V1'] = '0'

MODEL_PATH = '/mnt/data/huangjiawei/models/Baichuan-M2-32B'
OUT_DIR = '/mnt/data/huangjiawei/datasets_local/medical_dpo'
INP = os.path.join(OUT_DIR, 'prompts_pool_43k.jsonl')
LOCAL_INP = '/tmp/prompts_pool_43k.jsonl'
BAD_IDS = os.path.join(OUT_DIR, 'all_bad_ids_for_smoke.txt')
LOCAL_BAD = '/tmp/all_bad_ids_for_smoke.txt'
STAGING_TMPL = os.path.join(OUT_DIR, 'chosen_rerun_bad_4307.jsonl.staging.shard{sid}')

NUM_GPUS = 8
BATCH = 16
QUEUE_MAXSIZE = NUM_GPUS * 4
MAX_MODEL_LEN = 8192
MAX_TOKENS = 6400


def worker(shard_id, queue, out_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(shard_id)
    from queue import Empty
    from vllm import LLM, SamplingParams

    print(f'[shard {shard_id}] loading vllm M2 TP=1 bf16, '
          f'max_model_len={MAX_MODEL_LEN}, max_tokens={MAX_TOKENS}...', flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.93,
        max_model_len=MAX_MODEL_LEN,
        dtype='bfloat16',
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAX_TOKENS)
    print(f'[shard {shard_id}] vllm ready in {time.time()-t0:.0f}s', flush=True)

    out = open(out_path, 'a', encoding='utf-8', buffering=1)  # 行缓冲
    done = 0
    cap_total = 0     # 撞顶累计
    t_first = time.time()
    while True:
        try:
            batch = queue.get(timeout=30)
        except Empty:
            print(f'[shard {shard_id}] queue empty timeout, exit', flush=True)
            break
        if batch is None:
            print(f'[shard {shard_id}] poison pill, exit', flush=True)
            break

        formatted = [b[3] for b in batch]
        try:
            outputs = llm.generate(formatted, sampling)
        except Exception as e:
            print(f'[shard {shard_id}] generate FAILED batch={len(batch)}: {e}', flush=True)
            continue

        cap_in_batch = 0
        for i, o in enumerate(outputs):
            item = batch[i]
            full = o.outputs[0].text
            fin = o.outputs[0].finish_reason
            out_tokens = len(o.outputs[0].token_ids)

            if fin == 'length':
                cap_in_batch += 1

            if '</think>' in full:
                parts = full.split('</think>', 1)
                thinking = parts[0].replace('<think>', '').strip()
                answer = parts[1].strip()
            else:
                thinking = ''
                answer = full.strip()

            out.write(json.dumps({
                'id': item[0],
                'prompt': item[1],
                'domain': item[2],
                'thinking': thinking,
                'chosen': answer,
                'teacher': 'baichuan-m2-32b',
                'finish_reason': fin,
                'output_tokens': out_tokens,
            }, ensure_ascii=False) + '\n')
        out.flush()
        done += len(batch)
        cap_total += cap_in_batch
        cap_pct = cap_in_batch / len(batch) * 100
        el = time.time() - t_first
        rate = done / el * 60 if el > 0 else 0
        msg = (f'[shard {shard_id}] +{len(batch)} done={done} '
               f'rate={rate:.1f}/min cap={cap_in_batch}/{len(batch)} ({cap_pct:.0f}%)')
        if cap_pct > 2.0:
            msg += '  ⚠ALERT'
        print(msg, flush=True)

    out.close()
    print(f'[shard {shard_id}] FINISHED done={done} cap_total={cap_total} '
          f'({cap_total/max(done,1)*100:.2f}%)', flush=True)


def main():
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # fallback pre-cp (launcher 应该已串行 cp 过)
    if not os.path.exists(LOCAL_INP):
        shutil.copy(INP, LOCAL_INP)
    if not os.path.exists(LOCAL_BAD):
        shutil.copy(BAD_IDS, LOCAL_BAD)

    # 1) 读 bad ids
    with open(LOCAL_BAD, encoding='utf-8') as f:
        bad_ids = set(int(x.strip()) for x in f if x.strip())
    print(f'[main] bad ids: {len(bad_ids)}', flush=True)

    # 2) resume: 读 staging 已 done
    done_ids = set()
    for sid in range(NUM_GPUS):
        sf = STAGING_TMPL.format(sid=sid)
        if os.path.exists(sf):
            with open(sf, encoding='utf-8') as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)['id'])
                    except Exception:
                        continue
    print(f'[main] resume done: {len(done_ids)}', flush=True)

    # 3) 加载 tokenizer + 准备 missing
    print(f'[main] loading tokenizer...', flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    missing = []
    domain_count = Counter()
    pool_total = 0
    with open(LOCAL_INP, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            pool_total += 1
            pid = r['id']
            if pid not in bad_ids:
                continue
            if pid in done_ids:
                continue
            text = tok.apply_chat_template(
                [{'role': 'user', 'content': r['prompt']}],
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode='on',
            )
            missing.append((pid, r['prompt'], r['domain'], text))
            domain_count[r['domain']] += 1

    total = len(missing)
    print(f'[main] pool={pool_total} bad={len(bad_ids)} resume={len(done_ids)} '
          f'missing={total} by_domain={dict(domain_count)}', flush=True)

    if total == 0:
        print(f'[main] all done, nothing to do', flush=True)
        return

    # 4) 起 workers
    queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
    procs = []
    for sid in range(NUM_GPUS):
        out_path = STAGING_TMPL.format(sid=sid)
        p = mp.Process(target=worker, args=(sid, queue, out_path))
        p.start()
        procs.append(p)
    print(f'[main] spawned {NUM_GPUS} workers, will feed {total} prompts '
          f'in batches of {BATCH}', flush=True)

    # 5) feeder
    t_start = time.time()
    feed_done = 0
    PRINT_EVERY = max(BATCH * 20, 320)
    next_report = PRINT_EVERY
    for i in range(0, total, BATCH):
        batch = missing[i:i + BATCH]
        queue.put(batch)
        feed_done += len(batch)
        if feed_done >= next_report or feed_done == total:
            el = time.time() - t_start
            print(f'[main] fed {feed_done}/{total} into queue, '
                  f'elapsed={el/60:.1f}m', flush=True)
            next_report = feed_done + PRINT_EVERY

    for _ in range(NUM_GPUS):
        queue.put(None)
    print(f'[main] all batches fed + poison pills, waiting...', flush=True)

    for p in procs:
        p.join()

    el_total = time.time() - t_start
    print(f'[main] all workers exited, total elapsed={el_total/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
