"""从 chosen_v4_filtered.jsonl 抽 1k smoke 子集
按文件来源比例抽样, 保证分布 ~ 全量"""
import json, random
from pathlib import Path
from collections import defaultdict

random.seed(42)
IN = Path("/mnt/data/huangjiawei/datasets_local/medical_v4/chosen_v4_filtered.jsonl")
OUT = Path("/mnt/data/huangjiawei/datasets_local/medical_v4/smoke_1k.jsonl")

TARGET = 1000

# 读全部 + 按 src_file 分组
by_src = defaultdict(list)
with IN.open(encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        by_src[x["src_file"]].append(x)

print("[分布]")
total = sum(len(v) for v in by_src.values())
for k, v in by_src.items():
    print(f"  {k}: {len(v)}")

# 按比例抽
out = []
for src, items in by_src.items():
    n = round(TARGET * len(items) / total)
    sample = random.sample(items, min(n, len(items)))
    out.extend(sample)
    print(f"  -> {src}: {len(sample)}")

random.shuffle(out)
out = out[:TARGET]

with OUT.open("w", encoding="utf-8") as f:
    for x in out:
        # 重写 id 让 smoke 可独立
        x["id"] = f"smoke_{out.index(x):04d}"
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print(f"\n[saved] {len(out)} 条 → {OUT}")
