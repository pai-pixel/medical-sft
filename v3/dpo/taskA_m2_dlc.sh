#!/usr/bin/env bash
# Task A DLC launcher v2: M2-32B 生成 TCM 18K chosen
# 4 个 TP=2 vllm 实例并行 (4-shard split, 各占 2 卡)
# 资源池: PerceptiveMemory, 1 worker × 8 GPU
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[taskA-launcher] ulimit -n = $(ulimit -n)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0

PYBIN=/mnt/data/huangjiawei/_envs/vllm_env/bin/python
SCRIPT=/mnt/data/huangjiawei/scripts/10_taskA_m2_chosen.py
LOG_DIR=/mnt/data/huangjiawei/logs
mkdir -p "$LOG_DIR"

[ -x "$PYBIN" ] || { echo "[FATAL] MISS PYBIN"; exit 1; }
[ -f "$SCRIPT" ] || { echo "[FATAL] MISS SCRIPT"; exit 1; }
[ -f /mnt/data/huangjiawei/datasets_local/medical_dpo/prompts_pool_43k.jsonl ] || \
    { echo "[FATAL] MISS prompts pool"; exit 1; }
[ -d /mnt/data/huangjiawei/models/Baichuan-M2-32B ] || \
    { echo "[FATAL] MISS M2 model"; exit 1; }

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
echo "===================================================="
echo " Task A v2: M2-32B 生成 TCM 18K chosen"
echo " 4 个 TP=2 vllm 实例并行, 每实例占 2 卡"
echo " RUN_TAG: ${RUN_TAG}"
echo "===================================================="

# 关键修复: 串行先 cp prompts 到 /tmp, 防止 4 shard 并发 cp race condition
LOCAL_INP=/tmp/prompts_pool_43k.jsonl
if [ ! -f "$LOCAL_INP" ] || [ "$(stat -c%s "$LOCAL_INP")" -lt 5000000 ]; then
    echo "[launcher] pre-cp prompts to /tmp..."
    cp /mnt/data/huangjiawei/datasets_local/medical_dpo/prompts_pool_43k.jsonl "$LOCAL_INP"
    echo "[launcher] cp done, size=$(stat -c%s "$LOCAL_INP")"
fi

# 启 4 个 shard, 各占 GPU pair
declare -a PIDS
for shard in 0 1 2 3; do
    GPU_LO=$((shard * 2))
    GPU_HI=$((shard * 2 + 1))
    LOG="$LOG_DIR/taskA_m2_${RUN_TAG}_shard${shard}.log"
    echo "[launcher] starting shard $shard on GPU ${GPU_LO},${GPU_HI} -> $LOG"
    CUDA_VISIBLE_DEVICES=${GPU_LO},${GPU_HI} \
        $PYBIN $SCRIPT --shard=$shard --num_shards=4 --tp=2 \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "[launcher] all 4 shards started, PIDs: ${PIDS[@]}"
echo "[launcher] waiting for completion..."

# 等所有 shard 完成
EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[launcher] shard pid=$pid FAILED"
        EXIT_CODE=1
    fi
done

echo "[launcher] all shards finished, exit=$EXIT_CODE"

# 合并 staging shards 到 final
$PYBIN -c "
import os, json, shutil
from collections import Counter
OUT_DIR = '/mnt/data/huangjiawei/datasets_local/medical_dpo'
OUT = os.path.join(OUT_DIR, 'chosen_m2_18k.jsonl')
results, seen = [], set()
for shard in range(4):
    sf = os.path.join(OUT_DIR, f'chosen_m2.jsonl.staging.shard{shard}')
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
tmp = '/tmp/chosen_m2_final.jsonl'
with open(tmp, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
shutil.copy(tmp, OUT)
print(f'merged {len(results)} -> {OUT}')
print('domain dist:', dict(Counter(r['domain'] for r in results)))
"

exit $EXIT_CODE
