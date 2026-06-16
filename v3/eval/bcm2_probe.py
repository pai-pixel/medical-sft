import os, sys
# HF env MUST be set before importing huggingface_hub (it reads env at import time)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # XET silently stalls -> off
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")   # hf_transfer bypasses endpoint -> off
# do NOT set HF_ENDPOINT to hf-mirror; this node connects to hf.co directly and faster

from huggingface_hub import HfApi, hf_hub_download

REPO = sys.argv[1] if len(sys.argv) > 1 else "baichuan-inc/Baichuan-M2-32B"
api = HfApi()
try:
    info = api.model_info(REPO, files_metadata=True)
except Exception as e:
    print("MODEL_INFO_ERROR for %r: %r" % (REPO, e))
    sys.exit(1)

print("=== REPO:", REPO, "===")
print("pipeline_tag:", info.pipeline_tag)
print("library:", getattr(info, "library_name", None))
print("tags:", info.tags)
sib = info.siblings or []
tot = sum((s.size or 0) for s in sib)
print("num_files:", len(sib), " total_size_GB:", round(tot / 1e9, 2))
print("--- weight + config files ---")
for s in sib:
    nm = s.rfilename
    keep = nm.endswith((".safetensors", ".bin", ".json", ".model", ".txt", ".py", ".jinja")) or "README" in nm
    if keep:
        print("  %-44s %8.1f MB" % (nm, (s.size or 0) / 1e6))

def dump(fn, cap):
    try:
        p = hf_hub_download(REPO, fn, local_dir="/tmp/bcm2_meta")
        t = open(p, encoding="utf-8").read()
        print("\n=== %s (first %d chars) ===" % (fn, cap))
        print(t[:cap])
    except Exception as e:
        print("\n(%s fetch failed: %r)" % (fn, e))

dump("config.json", 4000)
dump("generation_config.json", 2000)
dump("README.md", 7000)
