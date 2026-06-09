#!/bin/bash
# Download ShenNong-TCM-Dataset (~113K TCM SFT instructions) to virtiofs.
# Run on hjw remote (huggingface_hub installed in default env).
#
# HF env MUST be exported BEFORE invoking hf CLI:
#   - HF_ENDPOINT=hf-mirror   (avoid CN GFW)
#   - HF_HUB_DISABLE_XET=1    (hf-mirror does not support XET)
#   - HF_HUB_ENABLE_HF_TRANSFER=0  (hf_transfer also bypasses endpoint)
# See memory: feedback_hf_disable_xet / feedback_hf_env_before_import.

set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=120

REPO_ID=michaelwzhu/ShenNong_TCM_Dataset
LOCAL_DIR=/mnt/data/huangjiawei/datasets_local/tcm/ShenNong-TCM-Dataset

mkdir -p "${LOCAL_DIR}"

echo "===================================================="
echo " repo   : ${REPO_ID}"
echo " local  : ${LOCAL_DIR}"
echo " endpoint=${HF_ENDPOINT}  xet=disabled  hf_transfer=disabled"
echo "===================================================="

hf download "${REPO_ID}" \
    --repo-type dataset \
    --local-dir "${LOCAL_DIR}" \
    --max-workers 4

echo
echo "===================================================="
echo " Files downloaded:"
echo "===================================================="
ls -lh "${LOCAL_DIR}"

echo
echo "===================================================="
echo " Preview first 2 records of each .jsonl / .json:"
echo "===================================================="
shopt -s nullglob
for f in "${LOCAL_DIR}"/*.jsonl "${LOCAL_DIR}"/*.json; do
    echo "--- ${f} ---"
    head -c 1200 "${f}"
    echo
    echo
done

echo
echo "Next step:"
echo "  python3 normalize_to_swift.py <input_file> ${LOCAL_DIR}/train_swift.jsonl"
