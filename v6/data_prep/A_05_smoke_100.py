"""A_05: smoke 100 抽样 → 用 opus_gen_answers.py 跑一遍验证 pipeline。

从 merged_prompts.jsonl 各 category 抽 25 条 = 100 条 smoke。
验证:
- 4 类 SYS prompt 都出 <think>...</think> + final
- 全 CN
- finish_reason 不撞顶 max_tokens=5120
"""
import json, random
from pathlib import Path
from collections import defaultdict

INP = Path("/mnt/data/huangjiawei/datasets_local/medical_v6/merged_prompts.jsonl")
OUT = Path("/mnt/data/huangjiawei/datasets_local/medical_v6/smoke_100.jsonl")

random.seed(42)
by_cat = defaultdict(list)
with INP.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        by_cat[r["category"]].append(r)

smoke = []
for cat in ["mcq", "acute", "tcm", "short"]:
    items = by_cat.get(cat, [])
    random.shuffle(items)
    smoke.extend(items[:25])

random.shuffle(smoke)
with OUT.open("w", encoding="utf-8") as f:
    for r in smoke:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
print(f"smoke 100: dist = {dict(Counter(r['category'] for r in smoke))}")
print(f"wrote {OUT}")
