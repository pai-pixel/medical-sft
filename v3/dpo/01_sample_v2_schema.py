"""Sample v2 train data to verify schema for DPO pipeline.
virtiofs hook: 必须 cp /tmp 后读 (2.1GB 全量 open 单线程会卡)
"""
import os, json, shutil, random
from collections import Counter

SRC = "/mnt/data/huangjiawei/datasets_local/medical_v2/train_v2.jsonl"
LOCAL = "/tmp/train_v2.jsonl"
N = 200

# Step 1: cp /tmp (200x faster than direct virtiofs scan)
if not os.path.exists(LOCAL):
    print("[cp] copying 2.1GB to /tmp ...", flush=True)
    shutil.copy(SRC, LOCAL)
    print("[cp] done", flush=True)

# Step 2: 随机抽 N 条 (reservoir sampling)
random.seed(42)
samples = []
with open(LOCAL, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < N:
            samples.append(line)
        else:
            j = random.randint(0, i)
            if j < N:
                samples[j] = line
print(f"[sample] read {i+1} lines, sampled {len(samples)}", flush=True)

# Step 3: parse 第一条看 schema
print("\n=== schema of row[0] ===", flush=True)
r0 = json.loads(samples[0])
for k, v in r0.items():
    if isinstance(v, str):
        print(f"  {k}: str len={len(v)} preview={v[:120]!r}")
    elif isinstance(v, list):
        print(f"  {k}: list len={len(v)}")
        if v and isinstance(v[0], dict):
            print(f"    elem[0] keys: {list(v[0].keys())}")
            for ek, ev in v[0].items():
                if isinstance(ev, str):
                    print(f"      {ek}: str preview={ev[:80]!r}")
                else:
                    print(f"      {ek}: {type(ev).__name__}")
    else:
        print(f"  {k}: {type(v).__name__} = {v!r}")

# Step 4: 统计 _track / _source 分布(快速看 domain 标签可用性)
tracks = Counter()
sources = Counter()
user_lens = []
for line in samples:
    r = json.loads(line)
    tracks[r.get("_track", "MISSING")] += 1
    sources[r.get("_source", "MISSING")] += 1
    msgs = r.get("messages", [])
    if msgs and msgs[0].get("role") == "user":
        user_lens.append(len(msgs[0].get("content", "")))

print(f"\n=== _track dist (200 samples) ===", flush=True)
for k, v in tracks.most_common():
    print(f"  {k}: {v}")

print(f"\n=== _source dist ===", flush=True)
for k, v in sources.most_common():
    print(f"  {k}: {v}")

print(f"\n=== user content len stats ===", flush=True)
if user_lens:
    user_lens.sort()
    print(f"  min={user_lens[0]} max={user_lens[-1]} mean={sum(user_lens)/len(user_lens):.0f}")
    print(f"  p10={user_lens[len(user_lens)//10]} p50={user_lens[len(user_lens)//2]} p90={user_lens[len(user_lens)*9//10]}")

# Step 5: 抽 3 条不同 _track 的 user content 展示
print("\n=== sample user prompts (per track) ===", flush=True)
shown = set()
for line in samples:
    r = json.loads(line)
    tr = r.get("_track", "?")
    if tr in shown: continue
    msgs = r.get("messages", [])
    if not msgs: continue
    user = msgs[0].get("content", "")
    print(f"\n--- {tr} (_source={r.get('_source')}) ---")
    print(user[:400])
    shown.add(tr)
    if len(shown) >= 4: break
