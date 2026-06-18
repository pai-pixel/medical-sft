"""Phase 1.2: 从 v2 train_v2.jsonl 抽 40K prompt (TCM 20K + EBM 20K).
过滤: 长度 [20,500] 字, 内容去重, 同模板(前40字)最多 50 条
"""
import os, json, hashlib, random, shutil
from collections import defaultdict, Counter

LOCAL = "/tmp/train_v2.jsonl"
OUT_DIR = "/mnt/data/huangjiawei/datasets_local/medical_dpo"
OUT = os.path.join(OUT_DIR, "prompts_v2_40k.jsonl")

TARGET = {"TCM": 20000, "EBM": 20000}
MIN_LEN, MAX_LEN = 20, 500
MAX_PER_TEMPLATE = 50

if not os.path.exists(LOCAL):
    SRC = "/mnt/data/huangjiawei/datasets_local/medical_v2/train_v2.jsonl"
    print("[cp] copying 2.1GB to /tmp ...", flush=True)
    shutil.copy(SRC, LOCAL)

seen_hash = set()
template_count = defaultdict(int)
pool_by_track = defaultdict(list)

total_read = filtered_short = filtered_long = filtered_dup = filtered_template = no_user = 0

with open(LOCAL, encoding="utf-8") as f:
    for line in f:
        total_read += 1
        try:
            r = json.loads(line)
        except Exception:
            continue
        track = r.get("_track")
        if track not in TARGET:
            continue
        msgs = r.get("messages", [])
        if not msgs or msgs[0].get("role") != "user":
            no_user += 1; continue
        user = msgs[0].get("content", "").strip()
        if len(user) < MIN_LEN:
            filtered_short += 1; continue
        if len(user) > MAX_LEN:
            filtered_long += 1; continue
        h = hashlib.md5(user.encode("utf-8")).hexdigest()
        if h in seen_hash:
            filtered_dup += 1; continue
        seen_hash.add(h)
        tpl = user[:40]
        if template_count[tpl] >= MAX_PER_TEMPLATE:
            filtered_template += 1; continue
        template_count[tpl] += 1
        pool_by_track[track].append({
            "prompt": user,
            "domain": track,
            "source": r.get("_source", ""),
        })

print(f"\n=== filter stats ===", flush=True)
print(f"  total read         : {total_read}")
print(f"  no user msg        : {no_user}")
print(f"  filtered short<{MIN_LEN}: {filtered_short}")
print(f"  filtered long>{MAX_LEN}: {filtered_long}")
print(f"  filtered dup       : {filtered_dup}")
print(f"  filtered tmpl>{MAX_PER_TEMPLATE}: {filtered_template}")
print(f"  TCM pool           : {len(pool_by_track['TCM'])}")
print(f"  EBM pool           : {len(pool_by_track['EBM'])}")

random.seed(42)
final = []
for track, target_n in TARGET.items():
    pool = pool_by_track[track]
    if len(pool) < target_n:
        print(f"WARN {track} pool too small: {len(pool)} < {target_n}", flush=True)
        final.extend(pool)
    else:
        final.extend(random.sample(pool, target_n))
random.shuffle(final)
print(f"\nfinal sampled: {len(final)}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
tmp_out = "/tmp/prompts_v2_40k.jsonl"
with open(tmp_out, "w", encoding="utf-8") as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
shutil.copy(tmp_out, OUT)
print(f"wrote -> {OUT}", flush=True)

dom_dist = Counter(r["domain"] for r in final)
src_dist = Counter(r["source"] for r in final)
len_dist = sorted(len(r["prompt"]) for r in final)
print(f"\n=== final domain dist ===", flush=True)
for k, v in dom_dist.most_common():
    print(f"  {k}: {v}")
print(f"\n=== final source dist (top 15) ===", flush=True)
for k, v in src_dist.most_common(15):
    print(f"  {k:30s}: {v}")
print(f"\n=== prompt len dist ===", flush=True)
print(f"  min={len_dist[0]} p10={len_dist[len(len_dist)//10]} p50={len_dist[len(len_dist)//2]} "
      f"p90={len_dist[len(len_dist)*9//10]} max={len_dist[-1]} mean={sum(len_dist)/len(len_dist):.0f}")
