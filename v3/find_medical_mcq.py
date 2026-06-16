import os, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import HfApi
from datasets import load_dataset

# CMMLU: plain parquet, fast. Peek TCM + clinical medical subjects.
for cfg in ["traditional_chinese_medicine", "clinical_knowledge", "college_medicine"]:
    print("==== cmmlu /", cfg, flush=True)
    try:
        ds = load_dataset("haonan-li/cmmlu", cfg, split="test", streaming=True)
        row = next(iter(ds))
        print("  keys:", list(row.keys()), flush=True)
        for k, v in row.items():
            print("    %-10s= %s" % (k, str(v)[:140]), flush=True)
    except Exception as e:
        print("  err:", repr(e)[:200], flush=True)

# CMB raw file list (avoid slow datasets-script resolution)
print("==== CMB raw files", flush=True)
try:
    info = HfApi().dataset_info("FreedomIntelligence/CMB", files_metadata=True)
    for s in info.siblings:
        sz = round((s.size or 0) / 1e6, 2) if s.size else "?"
        print("  %-55s %s MB" % (s.rfilename, sz), flush=True)
except Exception as e:
    print("  err:", repr(e)[:200], flush=True)
