#!/usr/bin/env bash
# Task B DLC launcher v2: Qwen3-8B 生成 43.5K rejected
# 8 个 TP=1 vllm 实例并行 (8-shard split, 各占 1 卡)
# 资源池: PerceptiveMemory, 1 worker × 8 GPU
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[taskB-launcher] ulimit -n = $(ulimit -n)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0

PYBIN=/mnt/data/huangjiawei/_envs/vllm_env/bin/python
SCRIPT=/mnt/data/huangjiawei/scripts/11_taskB_qwen3_rejected.py
LOG_DIR=/mnt/data/huangjiawei/logs
mkdir -p "$LOG_DIR"

[ -x "$PYBIN" ] || { echo "[FATAL] MISS PYBIN"; exit 1; }
[ -f "$SCRIPT" ] || { echo "[FATAL] MISS SCRIPT"; exit 1; }
[ -f /mnt/data/huangjiawei/datasets_local/medical_dpo/prompts_pool_43k.jsonl ] || \
    { echo "[FATAL] MISS prompts pool"; exit 1; }
[ -d /mnt/data/huangjiawei/models/Qwen3-8B ] || \
    { echo "[FATAL] MISS Qwen3-8B model"; exit 1; }

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
echo "===================================================="
echo " Task B v2: Qwen3-8B 生成 43.5K rejected"
echo " 8 个 TP=1 vllm 实例并行, 每实例占 1 卡"
echo " RUN_TAG: ${RUN_TAG}"
echo "===================================================="

declare -a PIDS
for shard in 0 1 2 3 4 5 6 7; do
    LOG="$LOG_DIR/taskB_qwen3_${RUN_TAG}_shard${shard}.log"
    echo "[launcher] starting shard $shard on GPU $shard -> $LOG"
    CUDA_VISIBLE_DEVICES=$shard \
        $PYBIN $SCRIPT --shard=$shard --num_shards=8 --tp=1 \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "[launcher] all 8 shards started, PIDs: ${PIDS[@]}"
echo "[launcher] waiting for completion..."

EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[launcher] shard pid=$pid FAILED"
        EXIT_CODE=1
    fi
done

echo "[launcher] all shards finished, exit=$EXIT_CODE"

# 合并 staging shards
$PYBIN -c "
import os, json, shutil
from collections import Counter
OUT_DIR = '/mnt/data/huangjiawei/datasets_local/medical_dpo'
OUT = os.path.join(OUT_DIR, 'rejected_qwen3_43k.jsonl')
results, seen = [], set()
for shard in range(8):
    sf = os.path.join(OUT_DIR, f'rejected_qwen3.jsonl.staging.shard{shard}')
    if not os.path.exists(sf):
        print(f'WARN missing {sf}')
        continue
    with open(sf, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                if r['id'] not in seen:
                    seen.add(r['id'])
                    results.append(r)
            except Exception:
                continue
tmp = '/tmp/rejected_qwen3_final.jsonl'
with open(tmp, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
shutil.copy(tmp, OUT)
print(f'merged {len(results)} -> {OUT}')
print('domain dist:', dict(Counter(r['domain'] for r in results)))
"

exit $EXIT_CODE
