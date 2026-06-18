#!/usr/bin/env bash
# Task D v2 launcher: M2-32B 跑 missing chosen
# 拓扑: 1 主 python + 8 worker × TP=1 + work-stealing queue
# (旧版 static sharding 因缺口 id 分布不均 stragglers 浪费 19 GPU-h, 重写)
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[taskD-v2-launcher] ulimit -n = $(ulimit -n)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0

PYBIN=/mnt/data/huangjiawei/_envs/vllm_latest/bin/python   # vllm 0.19.1
SCRIPT=/mnt/data/huangjiawei/scripts/12_taskD_m2_missing_chosen.py
LOG_DIR=/mnt/data/huangjiawei/logs
mkdir -p "$LOG_DIR"

[ -x "$PYBIN" ] || { echo "[FATAL] MISS PYBIN"; exit 1; }
[ -f "$SCRIPT" ] || { echo "[FATAL] MISS SCRIPT"; exit 1; }
[ -f /mnt/data/huangjiawei/datasets_local/medical_dpo/prompts_pool_43k.jsonl ] || \
    { echo "[FATAL] MISS prompts pool"; exit 1; }
[ -d /mnt/data/huangjiawei/models/Baichuan-M2-32B ] || \
    { echo "[FATAL] MISS M2 model"; exit 1; }

# 串行 pre-cp 三个 input 文件防 race (multi-shard cp /tmp 同文件踩过)
LOCAL_INP=/tmp/prompts_pool_43k.jsonl
if [ ! -f "$LOCAL_INP" ] || [ "$(stat -c%s "$LOCAL_INP")" -lt 5000000 ]; then
    echo "[launcher] pre-cp prompts_pool_43k to /tmp..."
    cp /mnt/data/huangjiawei/datasets_local/medical_dpo/prompts_pool_43k.jsonl "$LOCAL_INP"
fi
for src in chosen_m2_18k.jsonl chosen_opus_25k.jsonl; do
    sf=/mnt/data/huangjiawei/datasets_local/medical_dpo/$src
    df=/tmp/$src
    if [ -f "$sf" ]; then
        if [ ! -f "$df" ] || [ "$(stat -c%s "$df")" -lt 100000 ]; then
            echo "[launcher] pre-cp $src to /tmp..."
            cp "$sf" "$df"
        fi
    fi
done
echo "[launcher] pre-cp done"

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
MAIN_LOG="$LOG_DIR/taskD_m2_v2_${RUN_TAG}.log"

echo "===================================================="
echo " Task D v2: 8×TP=1 + work-stealing queue"
echo " RUN_TAG: ${RUN_TAG}"
echo " MAIN_LOG: ${MAIN_LOG}"
echo "===================================================="

# 主 python (内部 mp.spawn 8 workers); tee 到 log + stdout
$PYBIN -u "$SCRIPT" 2>&1 | tee "$MAIN_LOG" &
TEE_PID=$!

# launcher 60s monitor: 汇总 staging 累计 + 增量速率 (DLC 控制台双视角)
(
    SHARDS=(0 1 2 3 4 5 6 7)
    last_total=-1
    last_t=$(date +%s)
    while true; do
        sleep 60
        if ! kill -0 $TEE_PID 2>/dev/null; then
            break
        fi
        ts=$(date +%H:%M:%S)
        line="[$ts]"
        total=0
        for s in "${SHARDS[@]}"; do
            sf=/mnt/data/huangjiawei/datasets_local/medical_dpo/chosen_taskd_m2.jsonl.staging.shard${s}
            n=0; [ -f "$sf" ] && n=$(wc -l < "$sf" 2>/dev/null)
            total=$((total + n))
            line="$line s${s}=${n}"
        done
        now=$(date +%s); dt=$((now - last_t))
        if [ $last_total -ge 0 ] && [ $dt -gt 0 ]; then
            inc=$((total - last_total))
            rate_per_min=$(awk "BEGIN{printf \"%.1f\", $inc * 60.0 / $dt}")
            line="$line | total=$total (+${inc}, ${rate_per_min}/min)"
        else
            line="$line | total=$total"
        fi
        echo "$line"
        last_total=$total; last_t=$now
    done
) &
MONITOR_PID=$!

# 等主 python (work-stealing queue 自动消化所有任务, 无 stragglers)
wait $TEE_PID || echo "[launcher] main python exited non-zero (vllm 0.19.1 finalize bug if any, ignored)"

kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# Validate staging 行数
echo "[launcher] all done, validating staging..."
EXIT_CODE=0
total_lines=0
for shard in 0 1 2 3 4 5 6 7; do
    sf=/mnt/data/huangjiawei/datasets_local/medical_dpo/chosen_taskd_m2.jsonl.staging.shard${shard}
    if [ -f "$sf" ]; then
        n=$(wc -l < "$sf")
        echo "  shard $shard: $n lines"
        total_lines=$((total_lines + n))
    else
        echo "  shard $shard: STAGING MISSING"
        EXIT_CODE=1
    fi
done
echo "[launcher] total staging lines: $total_lines (target ~23251)"
if [ "$total_lines" -lt 22000 ]; then
    echo "  ⚠ total too low (<22000)"
    EXIT_CODE=1
fi

# 合并 8 shard staging → final
$PYBIN -c "
import os, json, shutil
from collections import Counter
OUT_DIR = '/mnt/data/huangjiawei/datasets_local/medical_dpo'
OUT = os.path.join(OUT_DIR, 'chosen_taskd_m2_23k.jsonl')
results, seen = [], set()
for shard in range(8):
    sf = os.path.join(OUT_DIR, f'chosen_taskd_m2.jsonl.staging.shard{shard}')
    if not os.path.exists(sf):
        print(f'WARN missing {sf}'); continue
    with open(sf, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                if r['id'] not in seen:
                    seen.add(r['id']); results.append(r)
            except Exception: continue
tmp = '/tmp/chosen_taskd_final.jsonl'
with open(tmp, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
shutil.copy(tmp, OUT)
print(f'merged {len(results)} -> {OUT}')
print('domain:', dict(Counter(r['domain'] for r in results)))
"

# Phase 2 完整性校验
$PYBIN -c "
import os, json
from collections import Counter
OUT_DIR = '/mnt/data/huangjiawei/datasets_local/medical_dpo'
all_chosen_ids = set()
for fname in ('chosen_m2_18k.jsonl', 'chosen_opus_25k.jsonl', 'chosen_taskd_m2_23k.jsonl'):
    fp = os.path.join(OUT_DIR, fname)
    if not os.path.exists(fp): continue
    with open(fp, encoding='utf-8') as f:
        for line in f:
            try: all_chosen_ids.add(json.loads(line)['id'])
            except: continue

all_prompt_ids = set()
all_prompts = {}
with open(os.path.join(OUT_DIR, 'prompts_pool_43k.jsonl'), encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        all_prompt_ids.add(r['id'])
        all_prompts[r['id']] = r['domain']

missing = all_prompt_ids - all_chosen_ids
print(f'==== Phase 2 final ====')
print(f'  total prompts:       {len(all_prompt_ids)}')
print(f'  total chosen:        {len(all_chosen_ids)}')
print(f'  still missing:       {len(missing)}')
if missing:
    miss_dom = Counter(all_prompts[i] for i in missing)
    print(f'  missing by domain:   {dict(miss_dom)}')
"

exit $EXIT_CODE
