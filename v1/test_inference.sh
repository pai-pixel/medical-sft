#!/bin/bash
# Quick inference test on a smoke / full-run checkpoint.
# Usage: bash test_inference.sh [CKPT_PATH]
#   default CKPT = the smoke run from 2026-06-03 13:13.

set -euo pipefail

CKPT=${1:-/mnt/data/huangjiawei/sft_runs/qwen3vl-2b-tcm-shennong-20260603_131333-smoke/v0-20260603-131345/checkpoint-48}
TEST_DIR=/mnt/data/huangjiawei/tcm_sft/infer_test_$(date +%Y%m%d_%H%M%S)
mkdir -p "${TEST_DIR}"

# 5 representative TCM Q&A prompts (no assistant turn -- model needs to fill it)
cat > "${TEST_DIR}/test.jsonl" <<'EOF'
{"messages": [{"role": "user", "content": "小柴胡汤的组成和主治是什么？"}]}
{"messages": [{"role": "user", "content": "肝郁脾虚应该如何辨证论治？请说明常用方剂。"}]}
{"messages": [{"role": "user", "content": "请解释'见肝之病，知肝传脾，当先实脾'这句话的含义。"}]}
{"messages": [{"role": "user", "content": "我最近总是失眠多梦，口干舌燥，舌红少苔，应该如何辨证？"}]}
{"messages": [{"role": "user", "content": "六味地黄丸适合什么样的人服用？请说明组成和功效。"}]}
EOF

echo "===================================================="
echo " ckpt    : ${CKPT}"
echo " test_dir: ${TEST_DIR}"
echo "===================================================="

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

swift infer \
    --model "${CKPT}" \
    --infer_backend pt \
    --val_dataset "${TEST_DIR}/test.jsonl" \
    --result_path "${TEST_DIR}/result.jsonl" \
    --max_new_tokens 512 \
    --temperature 0.3 \
    --stream false 2>&1 | tee "${TEST_DIR}/infer.log"

echo
echo "===================================================="
echo " Pretty-printed Q & A:"
echo "===================================================="
python3 - <<PYEOF
import json, pathlib
p = pathlib.Path("${TEST_DIR}/result.jsonl")
for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    r = json.loads(line)
    msgs = r.get("messages", [])
    q = next((m.get("content","") for m in msgs if m.get("role")=="user"), "?")
    a = r.get("response") or next((m.get("content","") for m in msgs if m.get("role")=="assistant"), r.get("output",""))
    print(f"\n[Q{i}] {q}\n[A{i}] {a}\n" + "-"*60)
PYEOF
