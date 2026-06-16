#!/usr/bin/env bash
# GRPO + LoRA + vLLM colocate, DLC 1 节点 × 8 卡 全量 (基于 v2 SFT ckpt-13144 起跑)
#
# 用法:
#   1) DLC 任务执行命令填: bash /mnt/data/huangjiawei/scripts/grpo_mcq_dlc.sh
#   2) 镜像沿用 v2 SFT: ai-pai-acr-jieyue-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/ai_acr_namespace/zhiyue:vqa-clean_cuDNN-fixed
#   3) 节点配置: 1 worker × 8 GPU (A800-80G)
#
# 数据/模型路径全部已在 /mnt/data/huangjiawei/ 就位,DLC 透明读
set -euo pipefail

# === 防 fd 慢泄漏: launcher shell 顶上抬到硬上限 (子进程 setrlimit 兜底)
ulimit -n 262144 || ulimit -n 131072 || true
echo "[grpo-dlc] ulimit -n = $(ulimit -n)"

# === 离线 (防 swift 启动探 modelscope/HF 卡死)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

# === 路径
PYBIN=/mnt/data/huangjiawei/_envs/vllm_env/bin/python
MODEL=/mnt/data/huangjiawei/sft_runs/qwen3vl-8b-medical-v2-dlcvr3jt6wprv8tg/v0-20260608-183719/checkpoint-13144
DATA=/mnt/data/huangjiawei/datasets_local/medical_mcq/train.jsonl
PLUGIN=/mnt/data/huangjiawei/scripts/mcq_reward.py

RUN_TAG=$(hostname | sed 's/-master-.*//;s/-worker-.*//')
OUTPUT=/mnt/data/huangjiawei/sft_runs/qwen3vl-8b-grpo-mcq-${RUN_TAG}
LOG_DIR=/mnt/data/huangjiawei/logs
mkdir -p "${OUTPUT}" "${LOG_DIR}"

# === 前置检查 (任一缺失立即退出,不浪费 DLC 资源)
[ -x "${PYBIN}" ]              || { echo "[FATAL] MISS PYBIN=${PYBIN}"; exit 1; }
[ -d "${MODEL}" ]              || { echo "[FATAL] MISS MODEL=${MODEL}"; exit 1; }
[ -f "${DATA}" ]               || { echo "[FATAL] MISS DATA=${DATA}"; exit 1; }
[ -f "${PLUGIN}" ]             || { echo "[FATAL] MISS PLUGIN=${PLUGIN}"; exit 1; }

# === swift/trl/peft 自愈 (DLC 容器 ~/.local 一般空,脚本内装一次,~1-2 min)
# 注意: 不加 --no-deps,否则 modelscope/aiofiles 等运行时依赖会缺,swift import 直接挂
echo "[grpo-dlc] ensuring ms-swift==4.1.0 / trl==0.29.1 / peft + deps installed in --user site"
${PYBIN} -m pip install --user --quiet --root-user-action=ignore \
    ms-swift==4.1.0 trl==0.29.1 peft modelscope 2>&1 | tail -10 || true
${PYBIN} -c "import swift, trl, peft, modelscope; print('[grpo-dlc] swift', swift.__version__, '/ trl', trl.__version__, '/ modelscope', modelscope.__version__)" \
  || { echo "[FATAL] swift/trl/peft/modelscope not importable"; exit 1; }

# === DLC 注入的分布式 env (从 v2 SFT 同款,DLC 平台自动填)
export NNODES=${NNODES:-${WORLD_SIZE:-1}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

echo "===================================================="
echo " GRPO + LoRA + vLLM colocate (1 节点 × ${NPROC_PER_NODE} 卡 全量)"
echo " model    : ${MODEL}"
echo " data     : ${DATA}  ($(wc -l < ${DATA}) lines)"
echo " reward   : ${PLUGIN}"
echo " output   : ${OUTPUT}"
echo " world    : ${NNODES} nodes × ${NPROC_PER_NODE} gpu (rank=${NODE_RANK})"
echo " master   : ${MASTER_ADDR}:${MASTER_PORT}"
echo "===================================================="

# === GRPO 启动
# 注意 (2026-06-16 踩坑): swift.cli.main rlhf 自己会起 torch.distributed.run 来 spawn 多卡,
# 所以这里**不要再套外层 torchrun**,否则双层嵌套 = NPROC × NPROC 个进程互争 NCCL 端口。
# 关键开关 (本地 smoke 实测必加):
#   --vllm_enforce_eager true     跳过 vllm 编译期 fd 炸弹
#   --vllm_mode colocate          训练/rollout 共享 GPU
#   --vllm_gpu_memory_utilization 0.5  policy(~17G) + vllm KV(~28G) 并存
# DDP 8 卡: per_device_bs 4 × ga 1 × 8 卡 = global bs 32 (1067 条 / 32 = 33 step/epoch × 2 epoch ≈ 66 step)
# num_generations 8 → 每 step 256 rollout
${PYBIN} -m swift.cli.main rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --dataset "${DATA}" \
    --split_dataset_ratio 0 \
    --external_plugins "${PLUGIN}" \
    --reward_funcs mcq_acc mcq_format \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 2048 \
    --vllm_enforce_eager true \
    --max_completion_length 1024 \
    --temperature 1.0 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 2 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 3 \
    --output_dir "${OUTPUT}" \
    --report_to tensorboard \
    --gradient_checkpointing true \
    2>&1 | tee "${OUTPUT}/train.rank${NODE_RANK}.log"
