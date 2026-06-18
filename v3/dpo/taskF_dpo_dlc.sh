#!/usr/bin/env bash
# Phase 4 DPO: Qwen3-8B-Instruct + LoRA r=16, β=0.1, lr=5e-7
# 框架: ms-swift 4.1.0 (跟 v3 GRPO 同体系)
# 数据: 43,488 preference pairs (M2 中医 + Opus EBM/Gen chosen vs Qwen3-8B 自答 rejected)
#
# 注意 (踩坑):
#   1) swift.cli.main rlhf 自带 torchrun, **外层不再套** (feedback_swift_main_double_spawn.md)
#   2) DLC 容器 ~/.local 空, pip --user 自愈 swift/trl/peft (~1-2 min)
#   3) max_length=8192 cover smoke p99=5513 token chosen + ~1k prompt
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[dpo-dlc] ulimit -n = $(ulimit -n)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

PYBIN=/mnt/data/huangjiawei/_envs/vllm_env/bin/python   # GRPO 同款 swift 4.1.0 venv
MODEL=/mnt/data/huangjiawei/models/Qwen3-8B
DATA=/mnt/data/huangjiawei/datasets_local/medical_dpo/dpo_train_43k.jsonl

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
OUTPUT=/mnt/data/huangjiawei/sft_runs/qwen3-8b-dpo-medical-${RUN_TAG}
LOG_DIR=/mnt/data/huangjiawei/logs
mkdir -p "${OUTPUT}" "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/dpo_${RUN_TAG}.log"

[ -x "${PYBIN}" ]              || { echo "[FATAL] MISS PYBIN=${PYBIN}"; exit 1; }
[ -d "${MODEL}" ]              || { echo "[FATAL] MISS MODEL=${MODEL}"; exit 1; }
[ -f "${DATA}" ]               || { echo "[FATAL] MISS DATA=${DATA}"; exit 1; }

# === swift / trl / peft 自愈 ===
echo "[dpo-dlc] ensuring ms-swift==4.1.0 / trl / peft / modelscope installed in --user site"
${PYBIN} -m pip install --user --quiet --root-user-action=ignore \
    ms-swift==4.1.0 trl==0.29.1 peft modelscope 2>&1 | tail -5 || true
${PYBIN} -c "import swift, trl, peft, modelscope; print('[dpo-dlc] swift', swift.__version__, '/ trl', trl.__version__, '/ peft', peft.__version__)" \
    || { echo "[FATAL] swift/trl/peft/modelscope not importable"; exit 1; }

# === DLC 注入分布式 env (DLC 平台自动填) ===
export NNODES=${NNODES:-${WORLD_SIZE:-1}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

echo "===================================================="
echo " Phase 4 DPO — Qwen3-8B-Instruct + LoRA r=16"
echo " model    : ${MODEL}"
echo " data     : ${DATA}  ($(wc -l < ${DATA}) lines)"
echo " output   : ${OUTPUT}"
echo " main log : ${MAIN_LOG}"
echo " world    : ${NNODES} nodes × ${NPROC_PER_NODE} gpu (rank=${NODE_RANK})"
echo " hyper    : β=0.1 / lr=5e-7 / epoch=2 / max_length=8192"
echo "===================================================="

# === DPO 启动 ===
# swift.cli.main rlhf 内部 spawn torchrun, 这里**不要**外层 torchrun (会双层嵌套炸 NCCL)
# DDP 8 卡: per_device_bs 1 × ga 8 × 8 卡 = global bs 64 (43k / 64 ≈ 680 step/epoch × 2 epoch ≈ 1360 step)
${PYBIN} -m swift.cli.main rlhf \
    --rlhf_type dpo \
    --model "${MODEL}" \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --dataset "${DATA}" \
    --beta 0.1 \
    --rpo_alpha 0 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 8192 \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory true \
    --output_dir "${OUTPUT}" \
    --report_to none \
    2>&1 | tee "${MAIN_LOG}"

echo "[dpo-dlc] training exit"
echo "[dpo-dlc] output: ${OUTPUT}"
echo "[dpo-dlc] log:    ${MAIN_LOG}"

# 列出 ckpt
echo "=== checkpoints ==="
ls -lh "${OUTPUT}"/v0-*/checkpoint-* 2>/dev/null | tail -10
