"""通用 chosen 过滤: 支持 Task A/C/D 三套 staging.

判定: tokenize raw output, total tokens >= max_tokens - margin → 撞顶丢
- 自适应 schema: 有 thinking 字段 (M2) / 无 thinking (Opus 网关纯回复)
- 接收 --max-tokens 参数, 跟当时 SamplingParams 一致

用法:
  python 13_filter_chosen.py --input chosen_m2_18k.jsonl --max-tokens 4096 \\
                              --out-clean clean.jsonl --out-bad-ids bad.txt --out-bad-full bad_full.jsonl
"""
import argparse
import json
import time
from collections import Counter

from transformers import AutoTokenizer

MODEL = '/mnt/data/huangjiawei/models/Baichuan-M2-32B'  # 用 M2 tokenizer 给所有 task (中文 char-level 接近)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--max-tokens', type=int, required=True)
    ap.add_argument('--out-clean', required=True)
    ap.add_argument('--out-bad-ids', required=True)
    ap.add_argument('--out-bad-full', required=True)
    ap.add_argument('--margin', type=int, default=5,
                    help='tokens margin: total >= max_tokens - margin → cap')
    return ap.parse_args()


def main():
    args = parse_args()
    print(f'[filter] input={args.input}', flush=True)
    print(f'[filter] max_tokens={args.max_tokens}, margin={args.margin}', flush=True)
    threshold = args.max_tokens - args.margin

    print(f'[filter] loading tokenizer (M2)...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print(f'[filter] reading staging...', flush=True)
    all_recs = []
    with open(args.input, encoding='utf-8') as fp:
        for line in fp:
            try:
                all_recs.append(json.loads(line))
            except Exception:
                continue
    print(f'[filter] total raw: {len(all_recs)}', flush=True)

    # 去重
    seen = set()
    unique = []
    for r in all_recs:
        if r['id'] not in seen:
            seen.add(r['id'])
            unique.append(r)
    print(f'[filter] unique by id: {len(unique)} (dups: {len(all_recs)-len(unique)})',
          flush=True)

    has_thinking_field = any('thinking' in r for r in unique[:100])
    print(f'[filter] schema: has_thinking_field={has_thinking_field}', flush=True)

    clean, bad = [], []
    reasons = Counter()
    domains_bad = Counter()
    t0 = time.time()
    for i, r in enumerate(unique):
        if i and i % 5000 == 0:
            el = time.time() - t0
            print(f'[filter]   {i}/{len(unique)} ({el:.0f}s)', flush=True)

        # 重建 raw output 文本
        if has_thinking_field:
            think = r.get('thinking', '')
            chosen = r.get('chosen', '')
            if think:
                raw = '<think>\n' + think + '\n</think>\n\n' + chosen
            else:
                # 没收尾 </think>, 整个内容是被切的 thinking
                raw = '<think>\n' + chosen
        else:
            # Opus 网关回复, 纯 chosen 文本
            raw = r.get('chosen', '')

        n_toks = len(tok.encode(raw, add_special_tokens=False))

        is_bad = False
        reason = None

        # rule 1: M2 thinking 任务 — empty thinking + chosen 长 → A 桶
        if has_thinking_field and not r.get('thinking') and len(r.get('chosen', '')) > 500:
            is_bad = True
            reason = 'A_empty_think_long'

        # rule 2: tokens 撞顶
        elif n_toks >= threshold:
            is_bad = True
            reason = 'tokens_capped'

        if is_bad:
            r['_filter_reason'] = reason
            r['_total_tokens'] = n_toks
            bad.append(r)
            reasons[reason] += 1
            domains_bad[r.get('domain', '?')] += 1
        else:
            clean.append(r)

    el = time.time() - t0
    print(f'[filter] done in {el:.0f}s', flush=True)
    print(f'[filter] clean: {len(clean)}', flush=True)
    print(f'[filter] bad  : {len(bad)} ({len(bad)/len(unique)*100:.2f}%)', flush=True)
    print(f'[filter] reasons: {dict(reasons)}', flush=True)
    print(f'[filter] bad by domain: {dict(domains_bad)}', flush=True)

    # 写 clean
    with open(args.out_clean, 'w', encoding='utf-8') as fout:
        for r in clean:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[filter] wrote clean: {args.out_clean}', flush=True)

    # 写 bad ids
    with open(args.out_bad_ids, 'w', encoding='utf-8') as fout:
        for r in bad:
            fout.write(f'{r["id"]}\n')
    print(f'[filter] wrote bad ids: {args.out_bad_ids}', flush=True)

    # 写 bad full
    with open(args.out_bad_full, 'w', encoding='utf-8') as fout:
        for r in bad:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[filter] wrote bad full: {args.out_bad_full}', flush=True)


if __name__ == '__main__':
    main()
