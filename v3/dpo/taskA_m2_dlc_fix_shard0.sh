#!/usr/bin/env bash
# Task A 补丁: 单独跑 shard 0 (TCM id%4==0, 4457 条)
# Race condition 后补,占 GPU 0-1 即可
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export PYTHONUNBUFFERED=1 VLLM_USE_V1=0

PYBIN=/mnt/data/huangjiawei/_envs/vllm_env/bin/python
SCRIPT=/mnt/data/huangjiawei/scripts/10_taskA_m2_chosen.py

LOCAL_INP=/tmp/prompts_pool_43k.jsonl
if [ ! -f "$LOCAL_INP" ] || [ "$(stat -c%s "$LOCAL_INP")" -lt 5000000 ]; then
    cp /mnt/data/huangjiawei/datasets_local/medical_dpo/prompts_pool_43k.jsonl "$LOCAL_INP"
fi

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
LOG=/mnt/data/huangjiawei/logs/taskA_m2_${RUN_TAG}_shard0_补.log

echo "[fix-shard0] start, log=$LOG"
CUDA_VISIBLE_DEVICES=0,1 \
    $PYBIN $SCRIPT --shard=0 --num_shards=4 --tp=2 \
    > "$LOG" 2>&1

# 合并 final
$PYBIN -c "
import os, json, shutil
from collections import Counter
OUT_DIR = '/mnt/data/huangjiawei/datasets_local/medical_dpo'
OUT = os.path.join(OUT_DIR, 'chosen_m2_18k.jsonl')
results, seen = [], set()
for shard in range(4):
    sf = os.path.join(OUT_DIR, f'chosen_m2.jsonl.staging.shard{shard}')
    if not os.path.exists(sf):
        print(f'WARN missing {sf}'); continue
    with open(sf, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                if r['id'] not in seen:
                    seen.add(r['id']); results.append(r)
            except Exception: continue
tmp = '/tmp/chosen_m2_final.jsonl'
with open(tmp, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
shutil.copy(tmp, OUT)
print(f'merged {len(results)} -> {OUT}')
print('domain:', dict(Counter(r['domain'] for r in results)))
"
