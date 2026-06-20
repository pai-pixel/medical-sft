#!/usr/bin/env bash
# v4 C 数据生成 - DLC 4 节点 × 8 卡 = 32 卡
# 启动方式: DLC 控制台 Worker count=4, Worker GPU=8, vllm 镜像
#   bash /mnt/data/huangjiawei/scripts/C_gen_m2_4node.sh
#
# 修复点 vs 单节点版:
#   - max_tokens 2000 → 4096 (修 81% 截断)
#   - 节点间 prompts[RANK::WORLD_SIZE] 静态 stride 分片
#   - 节点内 8 卡 TP=1 + work-stealing queue (不变)
#   - 每节点输出 v4_C_chosen.rank{N}.jsonl, 后续 cat 合并
#
# ETA: 5000 题 / 32 卡 ≈ 单节点 53min / 4 ≈ 13-15 min + 加载 5-10 min = ~25 min
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[v4-C-4n] ulimit -n = $(ulimit -n)"

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
SCRIPT=/mnt/data/huangjiawei/scripts/v4_C_gen_4node.py
MODEL=/mnt/data/huangjiawei/models/Baichuan-M2-32B
A_DATA=/mnt/data/huangjiawei/datasets_local/medical_v4/chosen_v4_filtered.jsonl
LOG_DIR=/mnt/data/huangjiawei/logs
mkdir -p "$LOG_DIR"

[ -x "$PYBIN" ]    || { echo "[FATAL] MISS PYBIN"; exit 1; }
[ -f "$SCRIPT" ]   || { echo "[FATAL] MISS SCRIPT=$SCRIPT"; exit 1; }
[ -d "$MODEL" ]    || { echo "[FATAL] MISS MODEL"; exit 1; }
[ -f "$A_DATA" ]   || { echo "[FATAL] MISS A_DATA"; exit 1; }

# DLC env
NODE_RANK=${RANK:-${NODE_RANK:-0}}
WORLD=${WORLD_SIZE:-${NNODES:-4}}
RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')

MAIN_LOG="$LOG_DIR/v4_C_4n_${RUN_TAG}_rank${NODE_RANK}.log"

echo "==============================================="
echo " v4 C generation - 4 NODES × 8 GPU"
echo " python  : $PYBIN"
echo " script  : $SCRIPT"
echo " model   : $MODEL"
echo " rank    : $NODE_RANK / $WORLD"
echo " out     : /mnt/data/huangjiawei/datasets_local/medical_v4/v4_C_chosen.rank${NODE_RANK}.jsonl"
echo " log     : $MAIN_LOG"
echo "==============================================="

$PYBIN -u "$SCRIPT" 2>&1 | tee "$MAIN_LOG"

rc=${PIPESTATUS[0]}
echo "==============================================="
echo " v4-C-4n EXIT rc=$rc rank=$NODE_RANK at $(date)"
echo "==============================================="

# 仅 rank 0 监控其他节点 + 合并
if [ "$NODE_RANK" = "0" ]; then
    echo "[rank 0] 等待其他节点完成..."
    OUT_DIR=/mnt/data/huangjiawei/datasets_local/medical_v4
    # 等其他 rank 文件就绪 (用文件创建时间检测; 节点间静态均衡, 时间差不大)
    for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
        sleep 30
        n_files=$(ls "$OUT_DIR"/v4_C_chosen.rank*.jsonl 2>/dev/null | wc -l)
        n_lines=$(cat "$OUT_DIR"/v4_C_chosen.rank*.jsonl 2>/dev/null | wc -l)
        echo "[rank 0 merge-wait $try] files=$n_files lines=$n_lines"
        if [ "$n_files" -ge "$WORLD" ] && [ "$n_lines" -ge 4900 ]; then
            break
        fi
    done
    cat "$OUT_DIR"/v4_C_chosen.rank*.jsonl > "$OUT_DIR"/v4_C_chosen.jsonl
    echo "[rank 0] merged → $OUT_DIR/v4_C_chosen.jsonl ($(wc -l < $OUT_DIR/v4_C_chosen.jsonl) lines)"
fi

exit $rc
