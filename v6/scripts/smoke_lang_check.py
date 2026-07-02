"""检查 answer 里的英文残留(超过 3 词的英文短语)"""
import json, re

with open("/tmp/smoke.jsonl") as f:
    recs = [json.loads(line) for line in f]

# 匹配 3+ 个连续英文单词(排除单独药名如 INH / mg / ml)
EN_PHRASE_RE = re.compile(r"\b[A-Za-z][A-Za-z']+(?:\s+[A-Za-z][A-Za-z']+){2,}")

flagged = []
for r in recs:
    matches = EN_PHRASE_RE.findall(r["answer"])
    if matches:
        flagged.append((r["id"], r["category"], matches[:3]))

print("含 3+ 词英文短语的 answer 数: %d/%d" % (len(flagged), len(recs)))
for fid, cat, ms in flagged[:15]:
    print("  [%s] %s: %s" % (cat, fid, ms))

# 分类统计
from collections import Counter
c = Counter(x[1] for x in flagged)
print("\n分类分布: %s" % dict(c))

# 具体看 thinking 段
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
think_en = 0
for r in recs:
    m = THINK_RE.search(r["answer"])
    if not m: continue
    think = m.group(1)
    if EN_PHRASE_RE.search(think):
        think_en += 1

print("\nthinking 段有 3+ 词英文的: %d/%d" % (think_en, len(recs)))
