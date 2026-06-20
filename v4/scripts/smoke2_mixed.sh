#!/usr/bin/env bash
# v4 二次 smoke - 用 v4_train_smoke_2k.jsonl (混合 A+D1+C)
# 验证: 混合数据 schema/loss 健康下降
# 单节点 8 卡, ~10 min 跑完
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

PYBIN=/mnt/data/huangjiawei/vllm_env/bin/python
SWIFT_PKGS=/mnt/data/huangjiawei/swift_pkgs
MODEL=/mnt/data/huangjiawei/models/Qwen3-8B
DATA=/mnt/data/huangjiawei/datasets_local/medical_v4/v4_train_smoke_2k.jsonl
OUTPUT=/mnt/data/huangjiawei/sft_runs/v4_smoke2_mixed_$(date +%Y%m%d_%H%M%S)
LOG_DIR=/mnt/data/huangjiawei/logs/v4_smoke
mkdir -p "$OUTPUT" "$LOG_DIR"
LOG=$LOG_DIR/smoke2_$(date +%H%M%S).log

[ -f "$DATA" ] || { echo "[FATAL] MISS DATA=$DATA"; exit 1; }

export PYTHONPATH="${SWIFT_PKGS}:${PYTHONPATH:-}"

echo "===================================================="
echo " v4 smoke2 (mixed A+D1+C) @ $(date)"
echo " data: $DATA  ($(wc -l < $DATA) lines)"
echo " out : $OUTPUT"
echo " log : $LOG"
echo " hyper: full SFT / lr 5e-6 / 1 epoch / max_len 4096 / zero2"
echo "===================================================="

$PYBIN -m torch.distributed.run \
    --nproc_per_node 8 --nnodes 1 --master_port 29501 \
    -m swift.cli.sft \
    --model "$MODEL" \
    --tuner_type full \
    --dataset "$DATA" \
    --output_dir "$OUTPUT" \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_length 4096 \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --warmup_ratio 0.03 \
    --eval_strategy no \
    --report_to none \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "[v4-smoke2] EXIT rc=$rc at $(date)"
exit $rc
