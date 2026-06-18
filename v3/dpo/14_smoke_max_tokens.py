"""Smoke 50-100 条 bad prompts, max_tokens 故意拉满 (max_model_len-prompt-64),
看真实 output token 分布 + finish_reason 分布, 决定全量重跑 max_tokens.

跑在 hjw DSW 单卡, 不需要 DLC.

输出: stdout 直接打印 p50/p95/p99/max + finish_reason Counter, 推荐 max_tokens.
"""
import json
import os
import random
import time
from collections import Counter

import resource
_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(262144, _h), _h))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MODELSCOPE_OFFLINE'] = '1'
os.environ['VLLM_USE_V1'] = '0'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = '/mnt/data/huangjiawei/models/Baichuan-M2-32B'
OUT_DIR = '/mnt/data/huangjiawei/datasets_local/medical_dpo'
PROMPTS_FILE = f'{OUT_DIR}/prompts_pool_43k.jsonl'
BAD_IDS_FILE = os.environ.get('BAD_IDS_FILE',
                              f'{OUT_DIR}/all_bad_ids_for_smoke.txt')   # A+D 合并 bad

MAX_MODEL_LEN = 8192
SMOKE_N = int(os.environ.get('SMOKE_N', '50'))


def main():
    print(f'[smoke] loading bad ids from {BAD_IDS_FILE}...', flush=True)
    with open(BAD_IDS_FILE, encoding='utf-8') as f:
        bad_ids = set(int(x.strip()) for x in f if x.strip())
    print(f'[smoke] bad ids count: {len(bad_ids)}', flush=True)

    print(f'[smoke] loading prompts pool...', flush=True)
    pool = {}
    with open(PROMPTS_FILE, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            pool[r['id']] = r
    print(f'[smoke] pool: {len(pool)}', flush=True)

    targets = [pool[i] for i in bad_ids if i in pool]
    print(f'[smoke] target candidates: {len(targets)}', flush=True)

    random.seed(42)
    samp = random.sample(targets, min(SMOKE_N, len(targets)))
    print(f'[smoke] sampling {len(samp)} for smoke', flush=True)

    print(f'[smoke] loading tokenizer...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts_text = []
    prompt_token_lens = []
    for r in samp:
        text = tok.apply_chat_template(
            [{'role': 'user', 'content': r['prompt']}],
            tokenize=False, add_generation_prompt=True, thinking_mode='on',
        )
        prompts_text.append(text)
        prompt_token_lens.append(len(tok.encode(text, add_special_tokens=False)))
    plen_max = max(prompt_token_lens)
    plen_p99 = sorted(prompt_token_lens)[int(len(prompt_token_lens)*0.99)]
    print(f'[smoke] prompt tokens: max={plen_max} p99={plen_p99} '
          f'mean={sum(prompt_token_lens)/len(prompt_token_lens):.0f}', flush=True)

    smoke_max_tokens = MAX_MODEL_LEN - plen_max - 64
    print(f'[smoke] smoke max_tokens = {MAX_MODEL_LEN} - {plen_max} - 64 = {smoke_max_tokens}',
          flush=True)

    print(f'[smoke] loading vllm M2 TP=1 bf16, max_model_len={MAX_MODEL_LEN}...', flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.93,
        max_model_len=MAX_MODEL_LEN,
        dtype='bfloat16',
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    print(f'[smoke] vllm ready in {time.time()-t0:.0f}s', flush=True)

    sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=smoke_max_tokens)
    print(f'[smoke] generating {len(samp)} prompts...', flush=True)
    t1 = time.time()
    outputs = llm.generate(prompts_text, sampling)
    el = time.time() - t1
    print(f'[smoke] generated in {el:.0f}s ({len(samp)/el:.2f} prompts/s)', flush=True)

    out_lens = sorted(len(o.outputs[0].token_ids) for o in outputs)
    finish = Counter(o.outputs[0].finish_reason for o in outputs)
    n = len(out_lens)
    p50 = out_lens[n//2]
    p75 = out_lens[int(n*.75)]
    p90 = out_lens[int(n*.90)]
    p95 = out_lens[int(n*.95)]
    p99 = out_lens[int(n*.99)]
    om = out_lens[-1]

    print(f'')
    print(f'====== SMOKE RESULT ======')
    print(f'output tokens: p50={p50} p75={p75} p90={p90} p95={p95} p99={p99} max={om}')
    print(f'finish_reason: {dict(finish)}')

    # 推荐 max_tokens (向上取整到 256)
    recommend_raw = max(int(p99 * 1.15), om + 256)
    recommend = ((recommend_raw + 255) // 256) * 256
    upper_bound = MAX_MODEL_LEN - plen_max - 64
    if recommend > upper_bound:
        print(f'⚠ recommend {recommend} > upper_bound {upper_bound}, cap to {upper_bound}')
        recommend = upper_bound

    print(f'')
    print(f'推荐 max_tokens (全量重跑用): {recommend}')
    print(f'  公式: max(p99 * 1.15, max + 256), 向上 256 边界, cap to max_model_len-prompt-64')
    print(f'  当前 cap: {finish.get("length", 0)}/{n} = {finish.get("length", 0)/n*100:.1f}%')

    # 也打印几个示例 chosen 末尾, 看是不是真说完了
    print(f'')
    print(f'====== Sample tails (top 3 longest) ======')
    by_len = sorted(zip(samp, outputs, out_lens), key=lambda x: -x[2])
    for i, (r, o, l) in enumerate(by_len[:3]):
        text = o.outputs[0].text
        tail = text[-300:].replace(chr(10), '/N/')
        print(f'{i+1}. id={r["id"]} dom={r["domain"]} tokens={l} finish={o.outputs[0].finish_reason}')
        print(f'   tail: ...{tail}')


if __name__ == '__main__':
    main()
