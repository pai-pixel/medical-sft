"""Phase 4b: merge v4 37k + v6 new clean → v6_train_full.jsonl

输入:
  - v4: /mnt/data/huangjiawei/datasets_local/medical_v4/train/v4_train_full.jsonl (37k)
  - v6 new: /mnt/data/huangjiawei/datasets_local/medical_v6/v6_new_clean.jsonl

策略:
- v6 新数据优先(新的精心设计, 补短板)
- prompt SHA1 去重: v6 覆盖 v4 (若同一 prompt)
- 输出统一 schema: {id, system, messages}
"""
import json, hashlib, random
from pathlib import Path
from collections import Counter

V6_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v6")
V4_TRAIN = Path("/tmp/v4_train_full.jsonl")  # 已 cp 到 /tmp
V6_NEW = Path("/tmp/v6_new_clean.jsonl")
TRAIN = V6_DIR / "v6_train_full.jsonl"
SMOKE = V6_DIR / "v6_train_smoke_2k.jsonl"


def main():
    seen_sha = set()
    counts = Counter()
    dups = Counter()
    rows_v6 = []
    rows_v4 = []

    # v6 新数据优先
    with V6_NEW.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompt = r["messages"][0]["content"]
            sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
            if sha in seen_sha:
                dups["v6"] += 1
                continue
            seen_sha.add(sha)
            rec = {
                "id": r.get("id"),
                "system": r["system"],
                "messages": r["messages"],
            }
            rows_v6.append(rec)
            counts["v6"] += 1

    # v4 补(去重 v6 已有的)
    with V4_TRAIN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompt = r["messages"][0]["content"]
            sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
            if sha in seen_sha:
                dups["v4"] += 1
                continue
            seen_sha.add(sha)
            rec = {
                "id": r.get("id", f"v4_{counts['v4']:05d}"),
                "system": r["system"],
                "messages": r["messages"],
            }
            rows_v4.append(rec)
            counts["v4"] += 1

    # 合并 + 打乱
    all_rows = rows_v6 + rows_v4
    random.seed(42)
    random.shuffle(all_rows)

    with TRAIN.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    smoke = random.sample(all_rows, min(2000, len(all_rows)))
    with SMOKE.open("w", encoding="utf-8") as f:
        for r in smoke:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    print(f"kept by source: {dict(counts)}")
    print(f"dropped (dup): {dict(dups)}")
    print(f"total: {total}")
    print(f"smoke: {len(smoke)} → {SMOKE}")
    print(f"train: {total} → {TRAIN}")


if __name__ == "__main__":
    main()
