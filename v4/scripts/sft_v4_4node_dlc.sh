#!/usr/bin/env bash
# v4 SFT — Qwen3-8B base + full SFT, 4 节点 × 8 GPU = 32 卡
#
# 4 节点改动 vs 2 节点版:
#   - NNODES default 2 → 4
#   - global_bs 64 → 128 (1 × 4 × 32, 业界 8B SFT 常用)
#   - ETA 3.6h → 1.6-1.8h
#
# 启动: DLC 控制台 Worker count=4, Worker GPU=8, vllm 镜像
#   bash /mnt/data/huangjiawei/scripts/sft_v4_4node_dlc.sh
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[v4-sft-4n] ulimit -n = $(ulimit -n)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TRITON_LOG_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=warning

PYBIN=/mnt/data/huangjiawei/vllm_env/bin/python
SWIFT_PKGS=/mnt/data/huangjiawei/swift_pkgs
MODEL=/mnt/data/huangjiawei/models/Qwen3-8B
DATA=/mnt/data/huangjiawei/datasets_local/medical_v4/v4_train_full.jsonl

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
OUTPUT=/mnt/data/huangjiawei/sft_runs/qwen3-8b-sft-medical-v4-4n-${RUN_TAG}
LOG_DIR=/mnt/data/huangjiawei/logs
mkdir -p "${OUTPUT}" "${LOG_DIR}"

[ -x "${PYBIN}" ]              || { echo "[FATAL] MISS PYBIN=${PYBIN}"; exit 1; }
[ -d "${SWIFT_PKGS}/swift" ]   || { echo "[FATAL] MISS SWIFT_PKGS"; exit 1; }
[ -d "${MODEL}" ]              || { echo "[FATAL] MISS MODEL"; exit 1; }
[ -f "${DATA}" ]               || { echo "[FATAL] MISS DATA=${DATA}"; exit 1; }

DATA_LINES=$(wc -l < "${DATA}")
[ "${DATA_LINES}" -gt 30000 ]  || { echo "[FATAL] DATA too small: ${DATA_LINES} 行"; exit 1; }

export PYTHONPATH="${SWIFT_PKGS}:${PYTHONPATH:-}"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${WORLD_SIZE:-${NNODES:-4}}
NODE_RANK=${RANK:-${NODE_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}

GLOBAL_BS=$((NNODES * NPROC_PER_NODE * 4))   # 1 × ga 4 × 32 = 128
MAIN_LOG="${LOG_DIR}/sft_v4_4n_${RUN_TAG}_rank${NODE_RANK}.log"

echo "===================================================="
echo " v4 SFT (4 NODES) — Qwen3-8B + full SFT"
echo " swift    : swift_pkgs 4.2.3 + vllm_env"
echo " model    : ${MODEL}"
echo " data     : ${DATA} (${DATA_LINES} lines)"
echo " output   : ${OUTPUT}"
echo " main log : ${MAIN_LOG}"
echo " world    : ${NNODES} nodes × ${NPROC_PER_NODE} gpu = $((NNODES * NPROC_PER_NODE)) total"
echo " rank     : ${NODE_RANK} / ${NNODES}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo " hyper    : full SFT / lr=5e-6 / 2 epoch / global_bs=${GLOBAL_BS} (1×4×32) / max_len=4096 / zero3 / grad_ckpt=on"
echo " save     : every 200 step, total_limit=10 (防 v3 早期 ckpt 被覆盖)"
echo "===================================================="

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

echo "[v4-sft-4n] training exit (rank ${NODE_RANK})"
echo "[v4-sft-4n] output: ${OUTPUT}"
echo "[v4-sft-4n] log:    ${MAIN_LOG}"

if [ "${NODE_RANK}" = "0" ]; then
    echo "=== checkpoints (rank 0) ==="
    ls -lh "${OUTPUT}"/v0-*/checkpoint-* 2>/dev/null | tail -10
fi
