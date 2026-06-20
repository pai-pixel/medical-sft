#!/usr/bin/env bash
# v4 smoke 1k SFT - 单节点 8 卡 swift full SFT
# 抄 v3 DPO launcher 已跑通的环境配方:
#   PYBIN = vllm_env/bin/python (PIL 完整, 不被 swift_pkgs 污染)
#   PYTHONPATH = swift_pkgs (4.2.3, --tuner_type full 开关)
#   PYBIN -m torch.distributed.run -m swift.cli.sft
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[v4-smoke] ulimit -n = $(ulimit -n)"

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
DATA=/mnt/data/huangjiawei/datasets_local/medical_v4/smoke_1k.jsonl
OUTPUT=/mnt/data/huangjiawei/sft_runs/v4_smoke_$(date +%Y%m%d_%H%M%S)
LOG_DIR=/mnt/data/huangjiawei/logs/v4_smoke
mkdir -p "$OUTPUT" "$LOG_DIR"
LOG=$LOG_DIR/smoke_$(date +%H%M%S).log

[ -x "$PYBIN" ]            || { echo "[FATAL] MISS PYBIN=$PYBIN"; exit 1; }
[ -d "$SWIFT_PKGS/swift" ] || { echo "[FATAL] MISS SWIFT_PKGS"; exit 1; }
[ -d "$MODEL" ]            || { echo "[FATAL] MISS MODEL"; exit 1; }
[ -f "$DATA" ]             || { echo "[FATAL] MISS DATA=$DATA"; exit 1; }

export PYTHONPATH="${SWIFT_PKGS}:${PYTHONPATH:-}"

NPROC=8
echo "===================================================="
echo " v4 smoke 1k SFT @ $(date)"
echo " python  : $PYBIN"
echo " model   : $MODEL"
echo " data    : $DATA  ($(wc -l < $DATA) lines)"
echo " output  : $OUTPUT"
echo " log     : $LOG"
echo " world   : 1 node × $NPROC gpu"
echo " hyper   : full SFT / lr=5e-6 / 1 epoch / max_len=4096 / zero2"
echo "===================================================="

$PYBIN -m torch.distributed.run \
    --nproc_per_node "$NPROC" \
    --nnodes 1 \
    --master_port 29501 \
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
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --warmup_ratio 0.03 \
    --eval_strategy no \
    --report_to none \
    2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "===================================================="
echo " v4 smoke EXIT rc=$rc at $(date)"
echo " output: $OUTPUT"
ls -la "$OUTPUT" 2>/dev/null
echo "===================================================="
exit $rc
