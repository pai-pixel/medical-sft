"""smoke 100 抽样审 + 统计"""
import json, re
from collections import defaultdict

with open("/tmp/smoke.jsonl") as f:
    recs = [json.loads(line) for line in f]

by_cat = defaultdict(list)
for r in recs:
    by_cat[r["category"]].append(r)

for cat in ["mcq", "acute", "tcm", "short"]:
    r = by_cat[cat][0]
    fin = r["finish_reason"]
    ans = r["answer"]
    print("=" * 60)
    print("CATEGORY: %s  finish: %s  ans_len: %d" % (cat, fin, len(ans)))
    print("--- prompt ---")
    print(r["prompt"][:200])
    print("--- answer[:700] ---")
    print(ans[:700])
    print()

has_think = sum(1 for r in recs if "<think>" in r["answer"] and "</think>" in r["answer"])
avg_len = sum(len(r["answer"]) for r in recs) / len(recs)
cn_re = re.compile(r"[一-鿿]")
avg_cn = sum(sum(1 for c in r["answer"] if cn_re.match(c)) / max(len(r["answer"]), 1) for r in recs) / len(recs)
finish_dist = defaultdict(int)
for r in recs:
    finish_dist[r["finish_reason"]] += 1

print("=== 统计 ===")
print("含 <think>...</think>: %d/%d" % (has_think, len(recs)))
print("平均答长: %d 字符" % avg_len)
print("平均中文占比: %.1f%%" % (avg_cn * 100))
print("finish_reason 分布: %s" % dict(finish_dist))
