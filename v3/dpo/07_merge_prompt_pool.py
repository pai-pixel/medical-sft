"""Phase 1.4: 合并 v2 36K filtered + Opus 7.5K staging = ~43.5K prompt 池.
注: Opus ambig_tcm_vs_ebm/ambig_rare 2.5K 类 hang 跳过 (内容审核疑似),
   只保留 5 个 General 类: symptom 2000 + safety 1500 + refuse 1000 + family 1000 + drug 2000 = 7500.
domain 字段统一, 加 source 标签, 最终 shuffle 输出.
"""
import os, json, random, shutil, hashlib
from collections import Counter

OUT_DIR = "/mnt/data/huangjiawei/datasets_local/medical_dpo"
V2_IN = os.path.join(OUT_DIR, "prompts_v2_36k_filtered.jsonl")
OPUS_STAGING = os.path.join(OUT_DIR, "prompts_opus_gen.jsonl.staging")
OUT = os.path.join(OUT_DIR, "prompts_pool_43k.jsonl")

# Opus category → domain 映射
CATEGORY_DOMAIN = {
    "symptom": "General", "safety": "General", "refuse": "General",
    "family": "General", "drug": "General",
    "ambig_tcm_vs_ebm": "Ambig", "ambig_rare": "Ambig",
}

assert os.path.exists(V2_IN), f"missing {V2_IN}"
assert os.path.exists(OPUS_STAGING), f"missing {OPUS_STAGING}"

# v2 36K
v2_items = []
with open(V2_IN, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        v2_items.append({
            "prompt": r["prompt"],
            "domain": r["domain"],
            "source": "v2",
            "v2_source": r.get("source", ""),
            "quality_score": r.get("quality_score"),
        })
print(f"v2: {len(v2_items)} loaded", flush=True)

# Opus from staging (直接读 jsonl, 每行一个 item)
opus_items = []
with open(OPUS_STAGING, encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        cat = r.get("category")
        if cat not in CATEGORY_DOMAIN:
            continue
        opus_items.append({
            "prompt": r["prompt"],
            "domain": CATEGORY_DOMAIN[cat],
            "source": "opus_gen",
            "category": cat,
        })
print(f"opus (staging): {len(opus_items)} loaded", flush=True)

# 跨源 + 内部 dedup
seen = set()
merged = []
for r in v2_items + opus_items:
    h = hashlib.md5(r["prompt"].encode()).hexdigest()
    if h in seen: continue
    seen.add(h)
    merged.append(r)
print(f"after cross-dedup: {len(merged)} (dropped {len(v2_items)+len(opus_items)-len(merged)})", flush=True)

random.seed(42)
random.shuffle(merged)

# 添加全局唯一 id
for i, r in enumerate(merged):
    r["id"] = i

tmp_out = "/tmp/prompts_pool_46k.jsonl"
with open(tmp_out, "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
shutil.copy(tmp_out, OUT)
print(f"wrote -> {OUT}", flush=True)

# 分布统计
print(f"\n=== domain dist ===")
ddist = Counter(r["domain"] for r in merged)
for d, n in ddist.most_common():
    pct = n / len(merged) * 100
    print(f"  {d:10s}: {n:6d} ({pct:.1f}%)")

print(f"\n=== source dist ===")
sdist = Counter(r["source"] for r in merged)
for s, n in sdist.most_common():
    print(f"  {s}: {n}")

print(f"\n=== Opus category breakdown ===")
cdist = Counter(r.get("category") for r in merged if r["source"] == "opus_gen")
for c, n in cdist.most_common():
    print(f"  {c}: {n}")

print(f"\n=== v2 source breakdown (top 10) ===")
vdist = Counter(r.get("v2_source") for r in merged if r["source"] == "v2")
for v, n in vdist.most_common(10):
    print(f"  {v}: {n}")

# 长度分布
lens = sorted(len(r["prompt"]) for r in merged)
print(f"\n=== prompt 长度分布 ===")
print(f"  min={lens[0]} p10={lens[len(lens)//10]} p50={lens[len(lens)//2]} "
      f"p90={lens[len(lens)*9//10]} max={lens[-1]} mean={sum(lens)/len(lens):.0f}")

# 抽 3 条不同 domain 的样本
print(f"\n=== 样本 ===")
shown = set()
for r in merged:
    d = r["domain"]
    if d in shown: continue
    print(f"\n--- {d} (source={r['source']}) ---")
    print(f"  {r['prompt'][:200]}")
    shown.add(d)
    if len(shown) >= 4: break
