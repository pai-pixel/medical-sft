"""检查 phase3 staging 是否有截断"""
import json, re
from collections import Counter

path = "/tmp/opus_all.jsonl"

recs = []
with open(path) as f:
    for line in f:
        recs.append(json.loads(line))

print(f"total: {len(recs)}")

# finish_reason 分布
finish = Counter(r["finish_reason"] for r in recs)
print(f"finish_reason: {dict(finish)}")
length_cnt = finish.get("length", 0)
pct = length_cnt / len(recs) * 100 if recs else 0
print(f"截断率: {pct:.2f}% ({length_cnt}/{len(recs)})")
if pct > 2.0:
    print("!!! 撞顶率 > 2%, max_tokens 不够 !!!")
else:
    print("撞顶率 OK (< 2%)")

# answer 字符长度分布
lens = sorted(len(r["answer"]) for r in recs)
n = len(lens)
if n > 0:
    print(f"\nanswer 字符长度:")
    print(f"  p50={lens[n//2]}  p90={lens[int(n*0.9)]}  p95={lens[int(n*0.95)]}  p99={lens[int(n*0.99)]}  max={lens[-1]}")

# 中文字符近似 tokens (Anthropic BPE 中文约 1 char = 1.5 tokens)
CN_RE = re.compile(r"[一-鿿]")
def approx_tokens(s):
    cn = sum(1 for c in s if CN_RE.match(c))
    en_chars = len(s) - cn
    return int(cn * 1.5 + en_chars * 0.3)

tok_lens = sorted(approx_tokens(r["answer"]) for r in recs)
if tok_lens:
    print(f"\napprox tokens (BPE 估算):")
    print(f"  p50={tok_lens[n//2]}  p90={tok_lens[int(n*0.9)]}  p95={tok_lens[int(n*0.95)]}  p99={tok_lens[int(n*0.99)]}  max={tok_lens[-1]}")
    print(f"\nmax_tokens=5120 上限 → 现有 max approx {tok_lens[-1]} tokens, 占 {tok_lens[-1]/5120*100:.1f}%")

# 按 category 分
print(f"\n=== by category ===")
by_cat = {}
for r in recs:
    c = r["category"]
    by_cat.setdefault(c, []).append(r)
for cat, rs in by_cat.items():
    ls = sorted(len(r["answer"]) for r in rs)
    n2 = len(ls)
    fl = sum(1 for r in rs if r["finish_reason"] == "length")
    print(f"  {cat}: n={n2}  len_p95={ls[int(n2*0.95)] if n2 else 0}  len_max={ls[-1] if ls else 0}  length_cnt={fl}")
