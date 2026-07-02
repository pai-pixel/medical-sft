"""A_04: 合并所有 seed_prompts/*.jsonl → merged_prompts.jsonl。

输入(存在的都合):
- hf_cmb_train_*.jsonl         (mcq, HF)
- hf_medical_o1_acute_*.jsonl  (acute, HF)
- hf_short_qa_*.jsonl          (short, HF)
- d1_textbook_tcm_seed.jsonl   (tcm, D1 教材)
- opus_mcq_gen_gen.jsonl       (mcq_gen, Opus)
- opus_acute_gen.jsonl         (acute, Opus)
- opus_tcm_gen.jsonl           (tcm, Opus)
- opus_short_gen.jsonl         (short, Opus)

去重: prompt md5 hash + category 内去重(不同 category 允许 prompt 相似)
统一 schema: {"id","prompt","source","category","gt_answer","meta"}
category 规范化: mcq_gen → mcq

输出: merged_prompts.jsonl
     merge_stats.json (分布统计)
"""
import json, hashlib, random
from pathlib import Path
from collections import Counter, defaultdict

V6_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v6")
SEED_DIR = V6_DIR / "seed_prompts"
OUT = V6_DIR / "merged_prompts.jsonl"
STATS = V6_DIR / "merge_stats.json"

# category 规范化
CAT_MAP = {
    "mcq_gen": "mcq",
    "mcq": "mcq",
    "acute": "acute",
    "tcm": "tcm",
    "short": "short",
}


def main():
    all_items = []
    files = sorted(SEED_DIR.glob("*.jsonl"))
    print(f"[merge] found {len(files)} seed files", flush=True)

    src_dist = Counter()
    for fp in files:
        n = 0
        with fp.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cat_raw = r.get("category", "")
                cat = CAT_MAP.get(cat_raw, cat_raw)
                if not cat:
                    continue
                item = {
                    "id": r.get("id"),
                    "prompt": (r.get("prompt") or "").strip(),
                    "source": r.get("source", fp.stem),
                    "category": cat,
                    "gt_answer": r.get("gt_answer"),
                    "meta": r.get("meta", {}),
                }
                if not item["id"] or not item["prompt"]:
                    continue
                all_items.append(item)
                n += 1
        src_dist[fp.name] = n
        print(f"  {fp.name}: {n}", flush=True)

    # 去重 (prompt md5, 跨类去重防泄漏)
    seen = set()
    dedup = []
    dropped_dup = 0
    for it in all_items:
        h = hashlib.md5(it["prompt"].encode()).hexdigest()
        if h in seen:
            dropped_dup += 1
            continue
        seen.add(h)
        dedup.append(it)

    # 打乱
    random.seed(42)
    random.shuffle(dedup)

    with OUT.open("w", encoding="utf-8") as f:
        for it in dedup:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    cat_dist = Counter(x["category"] for x in dedup)
    src_type_dist = Counter(
        "hf" if x["source"].startswith(("FreedomIntelligence", "Suprit", "shibing624", "michaelwzhu", "medical_v4"))
        else "opus" if x["source"].startswith("opus")
        else "other"
        for x in dedup
    )

    stats = {
        "total_before_dedup": len(all_items),
        "total_after_dedup": len(dedup),
        "dropped_dup": dropped_dup,
        "by_source_file": dict(src_dist),
        "by_category": dict(cat_dist),
        "by_source_type": dict(src_type_dist),
    }
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[merge] {len(all_items)} raw → {len(dedup)} unique (dropped {dropped_dup})", flush=True)
    print(f"[merge] by category: {dict(cat_dist)}", flush=True)
    print(f"[merge] by source type: {dict(src_type_dist)}", flush=True)
    print(f"[merge] wrote {OUT}, stats → {STATS}", flush=True)


if __name__ == "__main__":
    main()
