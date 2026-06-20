"""下载 MedQA-CN 中文医考题
优先 FreedomIntelligence (跟 CMB 同源, 稳定) 或 bigbio
"""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_OFFLINE"] = "0"

import sys, time
from huggingface_hub import snapshot_download

OUT_ROOT = "/mnt/data/huangjiawei/datasets_local/eval"
os.makedirs(OUT_ROOT, exist_ok=True)

# 候选 (按优先级)
JOBS = [
    ("bigbio/med_qa", "dataset", f"{OUT_ROOT}/MedQA"),
]

for repo_id, repo_type, local_dir in JOBS:
    print(f"[{time.strftime('%H:%M:%S')}] -> {repo_id}", flush=True)
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=repo_id, repo_type=repo_type,
            local_dir=local_dir, max_workers=2,
        )
        print(f"[{time.strftime('%H:%M:%S')}] OK in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] FAIL: {e}", flush=True)

print("[done]", flush=True)
