#!/usr/bin/env bash
# Phase 4 DPO 2-NODE: Qwen3-8B-Instruct + LoRA, 2 节点 × 8 卡 = 16 GPU
# 跟 1 节点版同 hyperparam, 只减半 wall time:
#   - global_bs 保持 64 (2 节点 ga=4 × 16卡 = 64)
#   - lr/β/epoch/max_length 全部一致, 步数 1360 不变, wall time ~1.5-2.5h
#
# 环境: 抄 v2 SFT 已跑通 2 节点配方
#   - PYBIN = /mnt/data/huangjiawei/vllm_env/bin/python  (v2 SFT 同款, 不是 _envs/vllm_env)
#   - swift_pkgs 4.2.3 走 PYTHONPATH (用户修改版, 跟 v2 SFT 一致)
#   - trl/peft 用 pip --user 自愈 (DPO 比 SFT 多需 trl)
#
# 工程避坑:
#   1) 走外层 torchrun + swift.cli.rlhf (train entry, swift_pkgs 4.2.3 自带)
#      ⚠ 不用 swift.cli.main rlhf — launcher 包装会自启 torchrun, 多节点必炸 NCCL
#   2) PYTHONPATH=swift_pkgs 优先于 ~/.local 的 pip 装版本, 用 4.2.3 不用 4.1.0
set -euo pipefail

ulimit -n 262144 || ulimit -n 131072 || true
echo "[dpo-dlc-2n] ulimit -n = $(ulimit -n)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

# === 关 NCCL/Triton verbose log, 让 tqdm 进度条不被 INFO 刷屏 ===
# (踩坑: 默认 NCCL_DEBUG=INFO 每 step 刷十几行 "Connected all trees"/"Channel via P2P", 进度条看不见)
export NCCL_DEBUG=WARN
export TRITON_LOG_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=warning

PYBIN=/mnt/data/huangjiawei/vllm_env/bin/python   # v2 SFT 同款 venv
SWIFT_PKGS=/mnt/data/huangjiawei/swift_pkgs        # 4.2.3 用户修改版
MODEL=/mnt/data/huangjiawei/models/Qwen3-8B
DATA=/mnt/data/huangjiawei/datasets_local/medical_dpo/dpo_train_43k.jsonl

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
OUTPUT=/mnt/data/huangjiawei/sft_runs/qwen3-8b-dpo-medical-2n-${RUN_TAG}
LOG_DIR=/mnt/data/huangjiawei/logs
mkdir -p "${OUTPUT}" "${LOG_DIR}"

[ -x "${PYBIN}" ]              || { echo "[FATAL] MISS PYBIN=${PYBIN}"; exit 1; }
[ -d "${SWIFT_PKGS}/swift" ]   || { echo "[FATAL] MISS SWIFT_PKGS=${SWIFT_PKGS}"; exit 1; }
[ -d "${MODEL}" ]              || { echo "[FATAL] MISS MODEL=${MODEL}"; exit 1; }
[ -f "${DATA}" ]               || { echo "[FATAL] MISS DATA=${DATA}"; exit 1; }

# === swift 走 PYTHONPATH (优先于 ~/.local pip 版), trl/peft 自愈 ===
export PYTHONPATH="${SWIFT_PKGS}:${PYTHONPATH:-}"
echo "[dpo-dlc-2n] ensuring trl/peft/modelscope installed in --user (swift 走 swift_pkgs 4.2.3)"
${PYBIN} -m pip install --user --quiet --root-user-action=ignore \
    trl==0.29.1 peft modelscope 2>&1 | tail -5 || true
${PYBIN} -c "import swift, trl, peft, modelscope; print('[dpo-dlc-2n] swift', swift.__version__, '/ trl', trl.__version__, '/ peft', peft.__version__)" \
    || { echo "[FATAL] swift/trl/peft/modelscope not importable"; exit 1; }

# === 2 节点 DLC 注入分布式 env (DLC 平台自动填) ===
export NNODES=${NNODES:-${WORLD_SIZE:-2}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# rank-aware log (双节点都写自己的 log, stdout 都到 DLC 控制台)
MAIN_LOG="${LOG_DIR}/dpo_2n_${RUN_TAG}_rank${NODE_RANK}.log"

echo "===================================================="
echo " Phase 4 DPO (2 NODES) — Qwen3-8B + LoRA r=16"
echo " swift    : swift_pkgs 4.2.3 + vllm_env (v2 SFT 同款)"
echo " model    : ${MODEL}"
echo " data     : ${DATA}  ($(wc -l < ${DATA}) lines)"
echo " output   : ${OUTPUT}"
echo " main log : ${MAIN_LOG}"
echo " world    : ${NNODES} nodes × ${NPROC_PER_NODE} gpu = $((NNODES * NPROC_PER_NODE)) total"
echo " rank     : ${NODE_RANK} / ${NNODES}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo " hyper    : β=0.1 / lr=5e-7 / epoch=2 / global_bs=64 (1×4×16) / max_length=6144 / grad_ckpt=on"
echo "===================================================="

# === 启动: 外层 torchrun + swift.cli.rlhf (跟 v2 SFT 模板对称) ===
${PYBIN} -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    -m swift.cli.rlhf \
    --rlhf_type dpo \
    --model "${MODEL}" \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --dataset "${DATA}" \
    --dataset_num_proc 8 \
    --beta 0.1 \
    --rpo_alpha 0 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 6144 \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory true \
    --output_dir "${OUTPUT}" \
    --report_to none \
    2>&1 | tee "${MAIN_LOG}"

echo "[dpo-dlc-2n] training exit (rank ${NODE_RANK})"
echo "[dpo-dlc-2n] output: ${OUTPUT}"
echo "[dpo-dlc-2n] log:    ${MAIN_LOG}"

# 仅 rank 0 列 ckpt
if [ "${NODE_RANK}" = "0" ]; then
    echo "=== checkpoints (rank 0) ==="
    ls -lh "${OUTPUT}"/v0-*/checkpoint-* 2>/dev/null | tail -10
fi
