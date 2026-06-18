"""Task D 过滤: 把撞顶截断的挑出来, 留下干净的, 输出 bad_ids 待重跑.

判定规则 (按优先级):
1. empty thinking + chosen 长 → 撞顶在 thinking 阶段就被切 (A 桶)
2. total token (re-tokenized raw output) >= max_tokens - 5 → vllm 撞 max_tokens 上限
3. chosen 末尾在词/数字/单位中间被切 → 启发式 (现在先 token 法兜住, 这个备选)

输出:
- chosen_taskd_m2_clean.jsonl  : 干净留下的
- chosen_taskd_m2_bad_ids.txt  : 待重跑的 prompt id list (整数, 一行一个)
- chosen_taskd_m2_bad_full.jsonl : 被过滤的完整 record (审计用)
"""
import json
import time
from glob import glob
from collections import Counter

from transformers import AutoTokenizer

MODEL = '/mnt/data/huangjiawei/models/Baichuan-M2-32B'
OUT_DIR = '/mnt/data/huangjiawei/datasets_local/medical_dpo'
SHARDS = sorted(glob(f'{OUT_DIR}/chosen_taskd_m2.jsonl.staging.shard*'))
CLEAN = f'{OUT_DIR}/chosen_taskd_m2_clean.jsonl'
BAD_IDS = f'{OUT_DIR}/chosen_taskd_m2_bad_ids.txt'
BAD_FULL = f'{OUT_DIR}/chosen_taskd_m2_bad_full.jsonl'

MAX_TOKENS = 3072
LENGTH_THRESHOLD = MAX_TOKENS - 5  # 留 5 token margin


def main():
    print(f'[filter] loading tokenizer...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print(f'[filter] reading {len(SHARDS)} shards...', flush=True)
    all_recs = []
    for f in SHARDS:
        with open(f, encoding='utf-8') as fp:
            for line in fp:
                try:
                    all_recs.append(json.loads(line))
                except Exception:
                    continue
    print(f'[filter] total raw records: {len(all_recs)}', flush=True)

    # 去重
    seen = set()
    unique = []
    for r in all_recs:
        if r['id'] not in seen:
            seen.add(r['id'])
            unique.append(r)
    print(f'[filter] unique by id: {len(unique)} (dups: {len(all_recs)-len(unique)})',
          flush=True)

    clean, bad = [], []
    reasons = Counter()
    domains_bad = Counter()
    t0 = time.time()
    for i, r in enumerate(unique):
        if i and i % 5000 == 0:
            el = time.time() - t0
            print(f'[filter]   {i}/{len(unique)} processed ({el:.0f}s)', flush=True)

        # 重建 vllm raw output 的 token 数
        # M2 thinking: <think>\n{thinking}\n</think>\n{chosen}  (无 thinking 时整段是 thinking 没收尾)
        if r['thinking']:
            raw = '<think>\n' + r['thinking'] + '\n</think>\n\n' + r['chosen']
        else:
            raw = '<think>\n' + r['chosen']
        n_toks = len(tok.encode(raw, add_special_tokens=False))

        is_bad = False
        reason = None

        # rule 1: A 桶 — empty thinking + chosen 长
        if not r['thinking'] and len(r['chosen']) > 500:
            is_bad = True
            reason = 'A_empty_think_long'

        # rule 2: tokens 撞顶
        elif n_toks >= LENGTH_THRESHOLD:
            is_bad = True
            reason = 'tokens_capped'

        if is_bad:
            r['_filter_reason'] = reason
            r['_total_tokens'] = n_toks
            bad.append(r)
            reasons[reason] += 1
            domains_bad[r['domain']] += 1
        else:
            clean.append(r)

    el = time.time() - t0
    print(f'[filter] done in {el:.0f}s', flush=True)
    print(f'[filter] clean: {len(clean)}', flush=True)
    print(f'[filter] bad  : {len(bad)} ({len(bad)/len(unique)*100:.2f}%)', flush=True)
    print(f'[filter] reasons: {dict(reasons)}', flush=True)
    print(f'[filter] bad by domain: {dict(domains_bad)}', flush=True)

    # 写出 clean
    with open(CLEAN, 'w', encoding='utf-8') as fout:
        for r in clean:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[filter] wrote clean: {CLEAN}', flush=True)

    # 写出 bad ids (一行一个 int)
    with open(BAD_IDS, 'w', encoding='utf-8') as fout:
        for r in bad:
            fout.write(f'{r["id"]}\n')
    print(f'[filter] wrote bad ids: {BAD_IDS}', flush=True)

    # 写出 bad full (审计用)
    with open(BAD_FULL, 'w', encoding='utf-8') as fout:
        for r in bad:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[filter] wrote bad full: {BAD_FULL}', flush=True)


if __name__ == '__main__':
    main()
