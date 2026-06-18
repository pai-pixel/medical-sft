#!/usr/bin/env bash
# Task E launcher: M2-32B 重跑 4,307 条 bad chosen (A bad 747 + D bad 3,560)
# 配置: 8×TP=1 + work-stealing queue, max_model_len=8192, max_tokens=6400
# 写盘加 finish_reason + output_tokens 字段 (按 SOP)
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[taskE-launcher] ulimit -n = $(ulimit -n)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0

PYBIN=/mnt/data/huangjiawei/_envs/vllm_latest/bin/python   # vllm 0.19.1
SCRIPT=/mnt/data/huangjiawei/scripts/15_taskE_rerun_bad.py
LOG_DIR=/mnt/data/huangjiawei/logs
DATA_DIR=/mnt/data/huangjiawei/datasets_local/medical_dpo
mkdir -p "$LOG_DIR"

[ -x "$PYBIN" ] || { echo "[FATAL] MISS PYBIN"; exit 1; }
[ -f "$SCRIPT" ] || { echo "[FATAL] MISS SCRIPT"; exit 1; }
[ -f "$DATA_DIR/prompts_pool_43k.jsonl" ] || { echo "[FATAL] MISS prompts pool"; exit 1; }
[ -f "$DATA_DIR/all_bad_ids_for_smoke.txt" ] || { echo "[FATAL] MISS bad ids file"; exit 1; }
[ -d /mnt/data/huangjiawei/models/Baichuan-M2-32B ] || { echo "[FATAL] MISS M2 model"; exit 1; }

# 串行 pre-cp 防 race (multi-shard 同时 cp /tmp 同文件踩过)
LOCAL_INP=/tmp/prompts_pool_43k.jsonl
if [ ! -f "$LOCAL_INP" ] || [ "$(stat -c%s "$LOCAL_INP")" -lt 5000000 ]; then
    echo "[launcher] pre-cp prompts_pool_43k to /tmp..."
    cp "$DATA_DIR/prompts_pool_43k.jsonl" "$LOCAL_INP"
fi
LOCAL_BAD=/tmp/all_bad_ids_for_smoke.txt
if [ ! -f "$LOCAL_BAD" ]; then
    echo "[launcher] pre-cp bad ids to /tmp..."
    cp "$DATA_DIR/all_bad_ids_for_smoke.txt" "$LOCAL_BAD"
fi
echo "[launcher] pre-cp done"

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
MAIN_LOG="$LOG_DIR/taskE_rerun_${RUN_TAG}.log"

echo "===================================================="
echo " Task E rerun: 8×TP=1 + work-stealing queue"
echo " 4,307 条 bad chosen, max_model_len=8192, max_tokens=6400"
echo " RUN_TAG: ${RUN_TAG}"
echo " MAIN_LOG: ${MAIN_LOG}"
echo "===================================================="

# 主 python (内部 mp.spawn 8 workers)
$PYBIN -u "$SCRIPT" 2>&1 | tee "$MAIN_LOG" &
TEE_PID=$!

# launcher 60s monitor
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
            sf=$DATA_DIR/chosen_rerun_bad_4307.jsonl.staging.shard${s}
            n=0; [ -f "$sf" ] && n=$(wc -l < "$sf" 2>/dev/null)
            total=$((total + n))
            line="$line s${s}=${n}"
        done
        now=$(date +%s); dt=$((now - last_t))
        if [ $last_total -ge 0 ] && [ $dt -gt 0 ]; then
            inc=$((total - last_total))
            rate_per_min=$(awk "BEGIN{printf \"%.1f\", $inc * 60.0 / $dt}")
            line="$line | total=$total/4307 (+${inc}, ${rate_per_min}/min)"
        else
            line="$line | total=$total/4307"
        fi
        echo "$line"
        last_total=$total; last_t=$now
    done
) &
MONITOR_PID=$!

wait $TEE_PID || echo "[launcher] main python exited non-zero (vllm finalize bug if any, ignored)"

kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# Validate staging
echo "[launcher] all done, validating staging..."
EXIT_CODE=0
total_lines=0
for shard in 0 1 2 3 4 5 6 7; do
    sf=$DATA_DIR/chosen_rerun_bad_4307.jsonl.staging.shard${shard}
    if [ -f "$sf" ]; then
        n=$(wc -l < "$sf")
        echo "  shard $shard: $n lines"
        total_lines=$((total_lines + n))
    else
        echo "  shard $shard: STAGING MISSING"
        EXIT_CODE=1
    fi
done
echo "[launcher] total staging: $total_lines / 4307"
if [ "$total_lines" -lt 4100 ]; then
    echo "  ⚠ total too low (<4100)"
    EXIT_CODE=1
fi

# 合并 staging → final
$PYBIN -c "
import os, json, shutil
from collections import Counter
D = '$DATA_DIR'
OUT = os.path.join(D, 'chosen_rerun_bad_4307.jsonl')
results, seen = [], set()
for shard in range(8):
    sf = os.path.join(D, f'chosen_rerun_bad_4307.jsonl.staging.shard{shard}')
    if not os.path.exists(sf):
        print(f'WARN missing {sf}'); continue
    with open(sf, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                if r['id'] not in seen:
                    seen.add(r['id']); results.append(r)
            except Exception: continue
tmp = '/tmp/chosen_rerun_final.jsonl'
with open(tmp, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
shutil.copy(tmp, OUT)
print(f'merged {len(results)} -> {OUT}')
print('domain:', dict(Counter(r['domain'] for r in results)))
print('finish:', dict(Counter(r.get('finish_reason','?') for r in results)))
cap = sum(1 for r in results if r.get('finish_reason')=='length')
print(f'cap rate: {cap}/{len(results)} = {cap/max(len(results),1)*100:.2f}%')
"

# Phase 2 完整性校验 (clean 三件套 + rerun)
$PYBIN -c "
import os, json
from collections import Counter
D = '$DATA_DIR'
all_chosen_ids = set()
for fname in ('chosen_m2_18k_clean.jsonl', 'chosen_opus_25k_clean.jsonl',
              'chosen_taskd_m2_clean.jsonl', 'chosen_rerun_bad_4307.jsonl'):
    fp = os.path.join(D, fname)
    if not os.path.exists(fp): print(f'WARN missing {fname}'); continue
    cnt = 0
    with open(fp, encoding='utf-8') as f:
        for line in f:
            try:
                all_chosen_ids.add(json.loads(line)['id']); cnt += 1
            except: continue
    print(f'  {fname}: {cnt}')

all_prompt_ids = set()
all_prompts = {}
with open(os.path.join(D, 'prompts_pool_43k.jsonl'), encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        all_prompt_ids.add(r['id'])
        all_prompts[r['id']] = r['domain']

missing = all_prompt_ids - all_chosen_ids
print(f'==== Phase 2 final (clean + rerun) ====')
print(f'  total prompts:       {len(all_prompt_ids)}')
print(f'  total chosen:        {len(all_chosen_ids)}')
print(f'  still missing:       {len(missing)}')
if missing:
    miss_dom = Counter(all_prompts[i] for i in missing)
    print(f'  missing by domain:   {dict(miss_dom)}')
"

exit $EXIT_CODE
