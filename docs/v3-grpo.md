# v3 · GRPO 跑通记录 (2026-06-16)

## 工程结果

| 项 | 值 |
|---|---|
| 训练日期 | 2026-06-16 10:53 → 12:04 |
| DLC 任务 ID | `dlcze5vd6msi2quy` |
| 硬件 | 1 节点 × 8 GPU A800-80G(资源池 PerceptiveMemory) |
| 框架 | ms-swift 4.1.0 + trl 0.29.1 + peft + vLLM 0.17.1(colocate) |
| 起点 | v2 full-SFT `checkpoint-13144`(2026-06-08~10 训出) |
| 数据 | CMMLU 医学 8 subject,1067 train + 250 heldout |
| 算法 | GRPO + LoRA r=8 + RLVR(`mcq_acc` + `mcq_format`) |
| 超参 | lr 1e-6, bs 4×8=32, num_generations 8, max_completion 1024, 2 epoch |
| 全量 step | 532 step(`(1067 × 8) / 32 × 2 = 532`)|
| 总耗时 | **1h 11m** |
| Checkpoint | `checkpoint-450 / -500 / -532`(save_total_limit=3) |

## 效果结果

### 5 窗口均值(完全平,GRPO 没起作用)
```
step 1-50:    acc 0.690 | reward 1.657
step 50-150:  acc 0.690 | reward 1.662
step 150-300: acc 0.675 | reward 1.650
step 300-450: acc 0.679 | reward 1.654
step 450-532: acc 0.679 | reward 1.655   ← 跟起点完全一样
```
5 window 的 acc 差 1.5pp,完全在统计噪声内。**ckpt-532 LoRA 加载后能力 ≈ 起点 ckpt-13144**。

### 4 个失败原因(按严重度)

1. **kl ≈ 0.001 全程贴 0**:lr=1e-6 + LoRA r=8 + KL 约束 三件套合力把更新强度压到几乎为零,**模型本质上没在学**。
2. **frac_reward_zero_std 0.25-0.50**:1/3 到 1/2 的 prompt 组,8 个 generation 全对或全错 → 组内 advantage = 0 → **对训练零贡献**。GRPO 全靠组内方差,方差为零等于白跑。
3. **起跑分数已经接近天花板**:step 1 就 acc=0.72,v2 SFT 给的底子+CMMLU 难度配合下,起跑接近 reward 上限 1.0,剩 30% 多半是真不会的题 — RL 啃不动。
4. **mcq_format 起跑 0.97-1.0**:格式分这块毫无优化空间,信号几乎全压在 acc 上,但 acc 又难涨。

## 简历价值定调

**工程闭环 ✅ / 效果提升 ❌**:
- ✅ 可写:打通 GRPO+RLVR+LoRA+vLLM colocate+自定义 reward 插件 端到端 RL 闭环
- ❌ 不可写:RL 提升了模型准确率(没提升)

## 跟 v1/v2 的位置关系

| 阶段 | 状态 | 拍板 |
|---|---|---|
| **v1** (2B+LoRA SFT) | 已废 | 8 道方剂题全错,2026-06-09 弃 |
| **v2** (8B full SFT 168万) | 已冷冻 | 实测推理差,2026-06-11 冷冻封存 |
| **v3** (LoRA + GRPO + RLVR) | **本文档** | 工程跑通,RL 效果零,2026-06-16 收工 |

## 如果以后想"真做出 RL 提升" (3 选 1 或组合)

| 修法 | 改动 | 预估效果 | 成本 |
|---|---|---|---|
| **A 加猛火**(最简) | lr 1e-6 → 5e-6 + LoRA r=8→32 + KL coef 减半 | 中,可能 5-10% | 再跑 1h |
| **B 换更难数据**(根治零方差) | CMMLU → MedQA-USMLE 中文版 / 执业医师真题(起跑 acc 0.3-0.5,组内方差大) | 大 | 1h + 0.5h 数据 |
| **C 换更弱起点**(根治天花板) | 不接 v2,直接 Qwen3-VL-8B-Instruct 原版 GRPO | 大,acc 0.3→0.7 是合理空间 | 再跑 1h |

## 关键学到的事

1. **GRPO 不看 loss,看 reward**:loss 在 ±1e-3 震荡是正常,GRPO 的 advantage 是组内归一化的,期望就在 0 附近
2. **RL 起作用的前提**:起跑 acc 远低于天花板 + 组内 reward 方差大 + 更新强度足够(kl 不能太小)
3. **RLVR 数据要"难但不假"**:CMMLU 是难度低导致零方差,要换执业医师真题这种起跑分低的
4. **DLC swift.cli.main 不要套 torchrun**(详见根仓库 docs/lessons-learned.md 后续追加)
