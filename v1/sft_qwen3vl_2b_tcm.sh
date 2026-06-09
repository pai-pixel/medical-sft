#!/bin/bash
# Qwen3-VL-2B-Instruct SFT on ShenNong-TCM-Dataset (~113K records).
# Using VLM base as text-only LLM: data has no `images` / no <image> tags,
# so ViT stays frozen this round but its weights are preserved so a later
# stage-2 visual SFT (tongue / face diagnosis) can warm-start from this ckpt.
#
# Target: 2-node x 8-GPU DLC, ms-swift + DeepSpeed ZeRO-2.
# Estimated wall time: ~1.5-2h for 3 epoch (text-only is ~3x faster than VLM).

set -euo pipefail

# ============================================================
# Paths
# ============================================================
MODEL_PATH=/mnt/data/zhge/models/Qwen3-VL-2B-Instruct
DATA_ROOT=/mnt/data/huangjiawei/datasets_local/tcm/ShenNong-TCM-Dataset
RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=/mnt/data/huangjiawei/sft_runs/qwen3vl-2b-tcm-shennong-${RUN_TAG}

# Output of normalize_to_swift.py (run that first, see README in download script)
DATASETS=(
    "${DATA_ROOT}/train_swift.jsonl"
)

# ============================================================
# Smoke mode: 1K samples, ~5min on 8-GPU single node
#   Generate the smoke jsonl up front via:
#     python3 normalize_to_swift.py <raw> ${DATA_ROOT}/train_swift_smoke.jsonl --max-samples 1000
# ============================================================
if [ "${SMOKE:-0}" = "1" ]; then
    DATASETS=("${DATA_ROOT}/train_swift_smoke.jsonl")
    OUTPUT_DIR="${OUTPUT_DIR}-smoke"
    echo "[smoke mode] using 1K-sample subset"
fi

# ============================================================
# Pre-flight
# ============================================================
[ -d "${MODEL_PATH}" ] || { echo "ERROR: model not found: ${MODEL_PATH}"; exit 1; }
for f in "${DATASETS[@]}"; do
    [ -f "${f}" ] || { echo "ERROR: dataset not found: ${f}"; echo "Hint: run normalize_to_swift.py first."; exit 1; }
done
mkdir -p "${OUTPUT_DIR}"

# ============================================================
# Distributed env (DLC injects RANK / MASTER_ADDR / MASTER_PORT / WORLD_SIZE)
# ============================================================
export NNODES=${NNODES:-${WORLD_SIZE:-2}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# ============================================================
# wandb (export WANDB_API_KEY externally, e.g. via /mnt/data/huangjiawei/.config/wandb.env)
# ============================================================
export WANDB_PROJECT=${WANDB_PROJECT:-tcm-sft}
export WANDB_NAME=qwen3vl-2b-tcm-shennong-$(date +%m%d-%H%M)
# report_to controls metrics backend: 'none' (no extra deps) | 'wandb' | 'tensorboard'
REPORT_TO=${REPORT_TO:-none}

echo "===================================================="
echo " model      : ${MODEL_PATH}"
echo " datasets   : ${#DATASETS[@]} file(s)"
for f in "${DATASETS[@]}"; do echo "   - ${f}"; done
echo " output     : ${OUTPUT_DIR}"
echo " world      : ${NNODES} nodes x ${NPROC_PER_NODE} gpu, rank=${NODE_RANK}"
echo " master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "===================================================="

# ============================================================
# Environment: reuse vllm_env (cp from jiangman, conda env with py3.10
# + torch 2.10 + transformers 4.57.6 + qwen_vl_utils + vllm 0.17.1).
# Plus a separate --target install of ms-swift / deepspeed / decord / accelerate
# at swift_pkgs, injected via PYTHONPATH (so vllm_env stays untouched).
#
# SMOKE mode keeps system python (DSW pod has ms-swift globally).
# ============================================================
USE_VENV=${USE_VENV:-1}
if [ "${SMOKE:-0}" = "1" ]; then USE_VENV=0; fi

if [ "${USE_VENV}" = "1" ]; then
    PYBIN=${PYBIN:-/mnt/data/huangjiawei/vllm_env/bin/python}
    SWIFT_PKGS=${SWIFT_PKGS:-/mnt/data/huangjiawei/swift_pkgs}
    [ -x "${PYBIN}" ] || { echo "ERROR: python not found: ${PYBIN}"; exit 1; }
    [ -d "${SWIFT_PKGS}/swift" ] || { echo "ERROR: swift_pkgs not found: ${SWIFT_PKGS}"; exit 1; }
    export PYTHONPATH="${SWIFT_PKGS}:${PYTHONPATH:-}"
    echo "[setup] python      : ${PYBIN}"
    echo "[setup] py version  : $(${PYBIN} --version 2>&1)"
    echo "[setup] swift_pkgs  : ${SWIFT_PKGS}"
    LAUNCHER="${PYBIN} -m torch.distributed.run"
else
    echo "[setup] using system python (no venv)"
    if command -v torchrun >/dev/null 2>&1; then
        LAUNCHER="torchrun"
    else
        LAUNCHER="python3 -m torch.distributed.run"
    fi
fi
echo "[setup] launcher    : ${LAUNCHER}"

# ============================================================
# Train
#   per_device_batch=4 x grad_accum=4 x 16 GPU = global_batch 256
#   max_length=2048 enough for TCM Q&A (avg ~300-800 tokens), VLM 4096 not needed.
# ============================================================
${LAUNCHER} \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    -m swift.cli.sft \
    --model "${MODEL_PATH}" \
    --train_type full \
    --dataset "${DATASETS[@]}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 2048 \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to "${REPORT_TO}" \
    --run_name "${WANDB_NAME}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
