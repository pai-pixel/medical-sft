#!/usr/bin/env bash
# GRPO 练手: 在 v2 (Qwen3-VL-8B 医疗 SFT, ckpt-13144) 上用中文医疗 MCQ 跑 GRPO
# 用法:
#   smoke (抖参数错):  MAX_STEPS=3 bash grpo_mcq.sh
#   正式练手:          bash grpo_mcq.sh
set -e
cd /mnt/data/huangjiawei
ulimit -n 262144 || true   # 262144: 131072 仍 flaky(sympy import 偶崩),再加余量;仍远低于会崩的 1048576
echo "[grpo] effective ulimit -n = $(ulimit -n)"

PY=/mnt/data/huangjiawei/_envs/vllm_env/bin/python   # 复用 jiangman vllm(torch/vllm/transformers) + ~/.local 装的 swift/trl/peft
MODEL=/mnt/data/huangjiawei/sft_runs/qwen3vl-8b-medical-v2-dlcvr3jt6wprv8tg/v0-20260608-183719/checkpoint-13144
DATA=/mnt/data/huangjiawei/datasets_local/medical_mcq/train.jsonl
PLUGIN=/mnt/data/huangjiawei/scripts/mcq_reward.py
OUT=/mnt/data/huangjiawei/sft_runs/v2-grpo-mcq
mkdir -p logs "$OUT"

MAX_STEPS=${MAX_STEPS:--1}
GPU=${GPU:-0}
LOG=logs/grpo_mcq_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=$GPU $PY -m swift.cli.main rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --tuner_type lora --lora_rank 8 --lora_alpha 32 \
  --dataset "$DATA" \
  --split_dataset_ratio 0 \
  --external_plugins "$PLUGIN" \
  --reward_funcs mcq_acc \
  --num_generations 8 \
  --use_vllm true --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.5 \
  --vllm_max_model_len 2048 --vllm_enforce_eager true \
  --max_completion_length 1024 \
  --temperature 1.0 \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --num_train_epochs 2 \
  --max_steps "$MAX_STEPS" \
  --logging_steps 1 \
  --save_steps 100 --save_total_limit 2 \
  --output_dir "$OUT" \
  --report_to tensorboard 2>&1 | tee "$LOG"
