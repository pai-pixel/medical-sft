#!/bin/bash
# Qwen3-VL-8B full SFT v2: TCM + Western medicine combined, 168 万 samples, 2 epoch
# v2.1: 加 packing + dataloader workers,GPU 利用率 80% -> 95%+
set -euo pipefail

PYBIN=/mnt/data/huangjiawei/vllm_env/bin/python
SWIFT_PKGS=/mnt/data/huangjiawei/swift_pkgs
MODEL=/mnt/data/zhiyue-L3-TerminalPerceptiveMemory/models/Qwen3-VL-8B-Instruct
DATA=/mnt/data/huangjiawei/datasets_local/medical_v2/train_v2.jsonl

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
OUTPUT=/mnt/data/huangjiawei/sft_runs/qwen3vl-8b-medical-v2-${RUN_TAG}

[ -x "${PYBIN}" ]              || { echo "MISS PYBIN=$PYBIN"; exit 1; }
[ -d "${SWIFT_PKGS}/swift" ]   || { echo "MISS SWIFT_PKGS=$SWIFT_PKGS"; exit 1; }
[ -f "${MODEL}/config.json" ]  || { echo "MISS MODEL=$MODEL"; exit 1; }
[ -f "${DATA}" ]               || { echo "MISS DATA=$DATA"; exit 1; }
mkdir -p "${OUTPUT}"

export NNODES=${NNODES:-${WORLD_SIZE:-2}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export PYTHONPATH="${SWIFT_PKGS}:${PYTHONPATH:-}"

echo "===================================================="
echo " v2.1 full SFT — Qwen3-VL-8B (TCM+EBM, packing on)"
echo " model    : ${MODEL}"
echo " data     : ${DATA}  ($(wc -l < ${DATA} 2>/dev/null) lines)"
echo " output   : ${OUTPUT}"
echo " world    : ${NNODES} nodes x ${NPROC_PER_NODE} gpu (rank=${NODE_RANK})"
echo " master   : ${MASTER_ADDR}:${MASTER_PORT}"
echo "===================================================="

# 加速点:
# --dataloader_num_workers 8     8 个 worker 并行 tokenize/读盘
# --dataloader_pin_memory true   CPU->GPU 传输加速
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
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 4096 \
    --gradient_checkpointing true \
    --deepspeed zero3 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory true \
    --output_dir "${OUTPUT}" \
    --report_to none \
    2>&1 | tee "${OUTPUT}/train.rank${NODE_RANK}.log"
