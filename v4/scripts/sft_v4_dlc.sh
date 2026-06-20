#!/usr/bin/env bash
# v4 SFT — Qwen3-8B base + full SFT, 2 节点 × 8 GPU = 16 卡
#
# 环境配方 (复用 v3 DPO taskF_dpo_2node_dlc.sh 已跑通):
#   PYBIN = /mnt/data/huangjiawei/vllm_env/bin/python
#   swift_pkgs 4.2.3 走 PYTHONPATH (--tuner_type full, 不是 --train_type)
#   外层 torch.distributed.run + swift.cli.sft (不用 swift.cli.main)
#
# 关键改动 vs v3 DPO:
#   - swift.cli.rlhf → swift.cli.sft
#   - --tuner_type lora → --tuner_type full (full SFT)
#   - 移除 LoRA / β / rpo_alpha / rlhf_type
#   - lr 5e-7 → 5e-6 (SFT 比 DPO 高 10x, 跟 v2 同款)
#   - max_length 6144 → 4096 (smoke 验证够用, 节省 step time)
#   - save_total_limit 3 → 10 (吸取 v3 早期 ckpt 全被覆盖的教训)
#   - --deepspeed zero2 → zero3 (full SFT 8B 优化器状态必须 zero3)
#
# 启动: DLC 控制台 Worker count=2, Worker GPU=8, vllm 镜像
#   bash /mnt/data/huangjiawei/scripts/sft_v4_dlc.sh
#
# ETA: 42k × 2 epoch / global_bs=64 = 1313 step × ~10 s/step ≈ 3.6 h
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[v4-sft] ulimit -n = $(ulimit -n)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

# 关 NCCL/Triton verbose, 让 tqdm 进度条不被 INFO 刷屏
export NCCL_DEBUG=WARN
export TRITON_LOG_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=warning

PYBIN=/mnt/data/huangjiawei/vllm_env/bin/python
SWIFT_PKGS=/mnt/data/huangjiawei/swift_pkgs
MODEL=/mnt/data/huangjiawei/models/Qwen3-8B
DATA=/mnt/data/huangjiawei/datasets_local/medical_v4/v4_train_full.jsonl

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
OUTPUT=/mnt/data/huangjiawei/sft_runs/qwen3-8b-sft-medical-v4-2n-${RUN_TAG}
LOG_DIR=/mnt/data/huangjiawei/logs
mkdir -p "${OUTPUT}" "${LOG_DIR}"

# 校验关键路径
[ -x "${PYBIN}" ]              || { echo "[FATAL] MISS PYBIN=${PYBIN}"; exit 1; }
[ -d "${SWIFT_PKGS}/swift" ]   || { echo "[FATAL] MISS SWIFT_PKGS=${SWIFT_PKGS}"; exit 1; }
[ -d "${MODEL}" ]              || { echo "[FATAL] MISS MODEL=${MODEL}"; exit 1; }
[ -f "${DATA}" ]               || { echo "[FATAL] MISS DATA=${DATA}"; exit 1; }

DATA_LINES=$(wc -l < "${DATA}")
[ "${DATA_LINES}" -gt 30000 ]  || { echo "[FATAL] DATA too small: ${DATA_LINES} 行 (期望 ≥30k)"; exit 1; }

export PYTHONPATH="${SWIFT_PKGS}:${PYTHONPATH:-}"

# DLC 多节点 env (DLC 平台自动注入)
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${WORLD_SIZE:-${NNODES:-2}}
NODE_RANK=${RANK:-${NODE_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}

MAIN_LOG="${LOG_DIR}/sft_v4_2n_${RUN_TAG}_rank${NODE_RANK}.log"

echo "===================================================="
echo " v4 SFT (2 NODES) — Qwen3-8B + full SFT"
echo " swift    : swift_pkgs 4.2.3 + vllm_env (v3 DPO 同款)"
echo " model    : ${MODEL}"
echo " data     : ${DATA} (${DATA_LINES} lines)"
echo " output   : ${OUTPUT}"
echo " main log : ${MAIN_LOG}"
echo " world    : ${NNODES} nodes × ${NPROC_PER_NODE} gpu = $((NNODES * NPROC_PER_NODE)) total"
echo " rank     : ${NODE_RANK} / ${NNODES}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo " hyper    : full SFT / lr=5e-6 / 2 epoch / global_bs=64 (1×4×16) / max_len=4096 / zero3 / grad_ckpt=on"
echo " save     : every 200 step, total_limit=10 (防 v3 早期 ckpt 被覆盖)"
echo "===================================================="

# 启动: 外层 torchrun + swift.cli.sft
${PYBIN} -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    -m swift.cli.sft \
    --model "${MODEL}" \
    --tuner_type full \
    --dataset "${DATA}" \
    --dataset_num_proc 8 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --save_steps 200 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --max_length 4096 \
    --gradient_checkpointing true \
    --deepspeed zero3 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory true \
    --eval_strategy no \
    --output_dir "${OUTPUT}" \
    --report_to none \
    2>&1 | tee "${MAIN_LOG}"

echo "[v4-sft-2n] training exit (rank ${NODE_RANK})"
echo "[v4-sft-2n] output: ${OUTPUT}"
echo "[v4-sft-2n] log:    ${MAIN_LOG}"

# 仅 rank 0 列 ckpt
if [ "${NODE_RANK}" = "0" ]; then
    echo "=== checkpoints (rank 0) ==="
    ls -lh "${OUTPUT}"/v0-*/checkpoint-* 2>/dev/null | tail -10
fi
