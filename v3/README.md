# v3 · GRPO 强化学习微调

> 在 v2 SFT ckpt-13144 基础上,用 GRPO + LoRA + vLLM colocate 跑可验证奖励 (RLVR) 强化学习。

## 一句话结论

**工程闭环跑通 ✅(DLC 8 卡 1h 11m,532 step 全跑完);效果零(reward/acc 全程平稳无提升)❌**。
RL 没学到东西的原因:起跑就 acc 0.72(接近天花板)+ kl ≈ 0.001(更新强度太小)+ 25-50% 组内方差为零。

详见 `../docs/v3-grpo.md`。

## 算法配方

- 算法:GRPO (Group-Relative Policy Optimization)
- 起点:v2 full-SFT ckpt-13144(`Qwen3-VL-8B-Instruct + 168 万双轨 SFT`)
- Adapter:LoRA r=8, α=32
- Rollout 引擎:vLLM **colocate**(训练/采样共享 GPU,`enforce_eager=true` 跳 torch.compile)
- 数据:CMMLU 医学子集(8 个 subject)1067 train + 250 heldout
- 奖励:**规则验证器**(RLVR)
  - `mcq_acc`:正则抽「答案：X」exact match 标答,对=1 错=0
  - `mcq_format`:有「答案：X」格式 → 1,否则 0
- 超参:lr 1e-6, bs 4 × 8卡 = 32, num_generations 8, max_completion_length 1024, 2 epoch = 532 step

## 文件清单

| 文件 | 作用 |
|---|---|
| `mcq_reward.py` | 自定义 reward 插件(MCQAccuracy + MCQFormat,swift 4.1.0 `orms` 注册) |
| `prep_medical_mcq.py` | 从 HF `haonan-li/cmmlu` 拉医学 8 subject → train.jsonl / heldout.jsonl |
| `find_medical_mcq.py` | 探查 CMMLU 数据集 schema |
| `run_grpo.py` | fd-wrapper(本地 smoke 用,setrlimit RLIMIT_NOFILE → hard) |
| `grpo_mcq.sh` | **本地 hjw DSW** 单卡 smoke launcher(swift_pkgs+~/.local) |
| `grpo_mcq_dlc.sh` | **DLC 8 卡全量** launcher(swift.cli.main rlhf 不套 torchrun) |
| `eval/test_m2_inference.py` | Baichuan-M2-32B 7 题对照测试(thinking 模式) |
| `eval/download_baichuan_m2.py` | M2 HF snapshot_download(env 必须 import 前设) |
| `eval/bcm2_probe.py` | M2 仓库 metadata 探查 |

## DLC 启动

```bash
# 执行命令(沿用 v2 SFT 镜像,1 节点 × 8 GPU,资源池 PerceptiveMemory)
bash /mnt/data/huangjiawei/scripts/grpo_mcq_dlc.sh
```

## 三个救命设置(本地长 run 必加,DLC 内置就够)

```bash
ulimit -n 262144                  # transformers 4.57.6 fd 慢泄漏防御
--vllm_enforce_eager true         # 跳过 vllm 编译期 fd 炸弹
HF_HUB_OFFLINE=1 \                # 防 swift 启动探 modelscope/HF 卡死
  TRANSFORMERS_OFFLINE=1 \
  MODELSCOPE_OFFLINE=1
```

## 三个关键踩坑(教训沉淀)

1. **不要外层套 torchrun**:`swift.cli.main rlhf` 自己会启 `torch.distributed.run`,外层再套 = N×N 双层 spawn,NCCL 端口冲突崩。直接 `python -m swift.cli.main rlhf` 即可。
2. **pip install 不加 --no-deps**:swift 启动需要 `modelscope` 运行时依赖,加 `--no-deps` 会缺包崩。
3. **资源池必须 PerceptiveMemory**:VLM 起点(Qwen3-VL),跑 ModelSystem(LLM 池)vllm 必撞 `shm_broadcast`。

## 评估对照 (`eval/`)

跟开源医疗 SOTA **Baichuan-M2-32B** 做 7 题对照测试,结果详见 `../docs/baichuan-m2-comparison.md`。
方剂 4/4 全对 + 辨证施治 + 西医循证 + 安全边界 全过 — 印证 v2 "蒸馏数据 SFT 学不到真推理" 的判断。
