#!/usr/bin/env bash
# v4 评估 v2 - 5 模型 × 1345 题串行推理
# 单节点 hjw 8 卡, 每模型 ~10-15 min, 总 ~60-80 min
set -e

cd /mnt/data/huangjiawei
ulimit -n 1048576

EVAL=/mnt/data/huangjiawei/datasets_local/eval/eval_v2_1390.jsonl
OUT_DIR=/mnt/data/huangjiawei/datasets_local/eval/outputs_v2
LOG_DIR=/mnt/data/huangjiawei/logs/eval_v2
mkdir -p "$OUT_DIR" "$LOG_DIR"

PYTHON=/mnt/data/huangjiawei/_envs/vllm_latest/bin/python
INFER=/mnt/data/huangjiawei/scripts/infer_eval.py

V4_CKPT=/mnt/data/huangjiawei/sft_runs/qwen3-8b-sft-medical-v4-4n-dlc1b8oa2yc80h7w/v0-20260620-002608/checkpoint-584
BASE_QWEN=/mnt/data/huangjiawei/models/Qwen3-8B
DPO_RUN=/mnt/data/huangjiawei/sft_runs/qwen3-8b-dpo-medical-2n-dlcmigv7ub840khh/v0-20260619-053609
M2=/mnt/data/huangjiawei/models/Baichuan-M2-32B
V2_CKPT=/mnt/data/huangjiawei/sft_runs/qwen3vl-8b-medical-v2-dlcvr3jt6wprv8tg/v0-20260608-183719/checkpoint-13144

run_one() {
    local TAG="$1"; local BASE="$2"; local ADAPTER="$3"
    local OUT="$OUT_DIR/${TAG}.jsonl"
    local LOG="$LOG_DIR/infer_${TAG}_$(date +%H%M%S).log"
    if [ -f "$OUT" ] && [ "$(wc -l <"$OUT")" -ge 1300 ]; then
        echo "[skip] $TAG already done ($(wc -l <"$OUT") lines)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] >>> START $TAG"
    if [ -n "$ADAPTER" ]; then
        $PYTHON -u "$INFER" --base "$BASE" --adapter "$ADAPTER" \
                --tag "$TAG" --out "$OUT" \
                --eval-set "$EVAL" --enforce-eager > "$LOG" 2>&1
    else
        $PYTHON -u "$INFER" --base "$BASE" \
                --tag "$TAG" --out "$OUT" \
                --eval-set "$EVAL" --enforce-eager > "$LOG" 2>&1
    fi
    local rc=$?; local n=$(wc -l <"$OUT" 2>/dev/null || echo 0)
    echo "[$(date +%H:%M:%S)] <<< DONE  $TAG rc=$rc lines=$n"
}

echo "===================================================="
echo " v4 eval v2 — 5 模型 × 1345 题"
echo " $(date)"
echo "===================================================="

run_one "v4_sft_584"       "$V4_CKPT"   ""
run_one "qwen3_8b_base"    "$BASE_QWEN" ""
run_one "dpo_ckpt_1200"    "$BASE_QWEN" "$DPO_RUN/checkpoint-1200"
run_one "m2_32b"           "$M2"        ""

echo "[$(date +%H:%M:%S)] core 4 done"
ls -la "$OUT_DIR/"
