"""下载 Qwen3-8B 纯文本版 (~16GB bf16) 到 hjw 本地.
HF env 必须 import 前设置 (memory: feedback_hf_env_before_import.md).
不用 hf-mirror (memory: feedback_hfmirror_vs_hfco_direct.md, 实际 hf.co 直连快).
"""
import os, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen3-8B"
DST = "/mnt/data/huangjiawei/models/Qwen3-8B"
os.makedirs(DST, exist_ok=True)

print(f"[{time.strftime('%F %T')}] start {REPO} -> {DST} (max_workers=4, resumable)", flush=True)
attempt = 0
while True:
    attempt += 1
    try:
        p = snapshot_download(repo_id=REPO, local_dir=DST, max_workers=4)
        print(f"[{time.strftime('%F %T')}] DONE (attempt {attempt}) -> {p}", flush=True)
        break
    except Exception as e:
        print(f"[{time.strftime('%F %T')}] retry {attempt} after error: {repr(e)[:200]}", flush=True)
        time.sleep(min(60, 2 ** min(attempt, 6)))
