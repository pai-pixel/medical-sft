import os, sys, time
# HF env MUST be set before importing huggingface_hub (read once at import)
os.environ["HF_HUB_DISABLE_XET"] = "1"          # XET silently stalls -> off
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # hf_transfer bypasses endpoint -> off
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   # detect CloudFront stalls -> raise -> outer retry
# NO HF_ENDPOINT override: this PAI node reaches hf.co directly and faster than hf-mirror

from huggingface_hub import snapshot_download

REPO = "baichuan-inc/Baichuan-M2-32B"
DST = "/mnt/data/huangjiawei/models/Baichuan-M2-32B"
os.makedirs(DST, exist_ok=True)

print("[%s] start %s -> %s (max_workers=4, resumable)" % (time.strftime("%F %T"), REPO, DST), flush=True)
attempt = 0
while True:
    attempt += 1
    try:
        p = snapshot_download(repo_id=REPO, local_dir=DST, max_workers=4)
        print("[%s] DONE (attempt %d) -> %s" % (time.strftime("%F %T"), attempt, p), flush=True)
        break
    except Exception as e:
        print("[%s] retry %d after error: %r" % (time.strftime("%F %T"), attempt, e), flush=True)
        time.sleep(min(60, 2 ** min(attempt, 6)))
