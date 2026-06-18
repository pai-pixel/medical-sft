"""Phase 4 DPO 数据准备:
- 读 4 件套 chosen + rejected (43,500 对)
- 按 id 配对成 ms-swift DPO 期望格式
- inline filter Task E 12 条 finish=length 撞顶
- 拼 thinking + answer (Qwen3 学生学完整 thinking 模式)
- 输出: dpo_train_43k.jsonl  (一行一个 preference 样本)

ms-swift 4.1.0 DPO 数据格式:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<chosen full>"}],
   "rejected_response": "<rejected full>"}
"""
import json
import os
import random
from collections import Counter

D = '/mnt/data/huangjiawei/datasets_local/medical_dpo'
CHOSEN_FILES = [
    f'{D}/chosen_m2_18k_clean.jsonl',
    f'{D}/chosen_opus_25k_clean.jsonl',
    f'{D}/chosen_taskd_m2_clean.jsonl',
    f'{D}/chosen_rerun_bad_4307.jsonl',
]
REJ_FILE = f'{D}/rejected_qwen3_43k.jsonl'
OUT = f'{D}/dpo_train_43k.jsonl'

random.seed(42)


def wrap_response(thinking: str, content: str) -> str:
    """合成 chat 完整 assistant response.
    - 有 thinking 字段 -> 包 <think>...</think> tag (M2 thinking 模式 / Qwen3 thinking 模式)
    - 无 thinking -> 直接 content (Opus 网关纯回复)
    """
    if thinking and thinking.strip():
        return f'<think>\n{thinking.strip()}\n</think>\n\n{content.strip()}'
    return content.strip()


def main():
    print('=== reading chosen 4 files ===', flush=True)
    chosen = {}
    cap_filtered = 0
    for f in CHOSEN_FILES:
        cnt = 0
        cap = 0
        with open(f, encoding='utf-8') as fp:
            for line in fp:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cnt += 1
                # filter Task E 撞顶 (Task A/C/D 没 finish_reason 字段, .get 默认 None 不命中)
                if r.get('finish_reason') == 'length':
                    cap += 1
                    cap_filtered += 1
                    continue
                chosen[r['id']] = r
        print(f'  {os.path.basename(f)}: {cnt} (cap_filtered={cap})', flush=True)
    print(f'  chosen total: {len(chosen)}, total cap_filtered: {cap_filtered}', flush=True)

    print('=== reading rejected ===', flush=True)
    rejected = {}
    with open(REJ_FILE, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            rejected[r['id']] = r
    print(f'  rejected: {len(rejected)}', flush=True)

    print('=== pairing by id ===', flush=True)
    paired = []
    domain_count = Counter()
    teacher_count = Counter()
    skipped = 0
    for pid, c in chosen.items():
        if pid not in rejected:
            skipped += 1
            continue
        rj = rejected[pid]

        chosen_resp = wrap_response(c.get('thinking', ''), c.get('chosen', ''))
        rejected_resp = wrap_response(rj.get('thinking', ''), rj.get('rejected', ''))

        if not chosen_resp or not rejected_resp:
            skipped += 1
            continue

        paired.append({
            'messages': [
                {'role': 'user', 'content': c['prompt']},
                {'role': 'assistant', 'content': chosen_resp},
            ],
            'rejected_response': rejected_resp,
            # 元数据 (训练时 ms-swift 会忽略, 用于审计)
            '_id': pid,
            '_domain': c.get('domain', '?'),
            '_teacher': c.get('teacher', '?'),
        })
        domain_count[c.get('domain', '?')] += 1
        teacher_count[c.get('teacher', '?')] += 1

    print(f'  paired: {len(paired)} (skipped={skipped})', flush=True)
    print(f'  domain: {dict(domain_count)}', flush=True)
    print(f'  teacher: {dict(teacher_count)}', flush=True)

    # 长度统计 (粗略 char-level, 帮判断 max_length)
    chosen_lens = [len(p['messages'][1]['content']) for p in paired]
    rejected_lens = [len(p['rejected_response']) for p in paired]
    chosen_lens.sort()
    rejected_lens.sort()
    n = len(paired)
    print('=== 长度分布 (字符) ===', flush=True)
    print(f'  chosen   p50={chosen_lens[n//2]} p90={chosen_lens[int(n*.9)]} '
          f'p99={chosen_lens[int(n*.99)]} max={chosen_lens[-1]}', flush=True)
    print(f'  rejected p50={rejected_lens[n//2]} p90={rejected_lens[int(n*.9)]} '
          f'p99={rejected_lens[int(n*.99)]} max={rejected_lens[-1]}', flush=True)
    print('  (M2 tokenizer 中文 ~2 char/token, Qwen3 ~1.5 char/token, '
          '取中位数估 token: char/1.7)', flush=True)

    # shuffle + 写
    random.shuffle(paired)
    with open(OUT, 'w', encoding='utf-8') as fout:
        for p in paired:
            fout.write(json.dumps(p, ensure_ascii=False) + '\n')
    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f'=== wrote {OUT} ({size_mb:.1f} MB, {len(paired)} lines) ===', flush=True)


if __name__ == '__main__':
    main()
