# v4 SFT 规划 (2026-06-19 起草)

> 目标: 修 v3 失败 + 修 v2 "回避临床判断" 硬伤,产出可交付的离线终端医疗模型 (RK3588 32GB)

## 总览

| 项 | 决策 |
|---|---|
| 起点 | **Qwen3-8B base** (纯 LLM, 终端部署友好;不复用 v2 VLM) |
| 训练方式 | **full SFT** (`--tuner_type full`, swift 4.2.3) |
| 数据规模 | **35-40k** (A 32k + C 3-5k + D 1-2k) |
| 训练资源 | **2 节点 × 8 GPU = 16 卡 DLC** |
| 训练超参 | lr=5e-6, 2 epoch, max_length=4096, zero3 (沿用 v2 同款,但纯 LLM) |
| 节奏 | **1k smoke 先跑** → 全量 → 评估 |
| 评估 | 复用 v3 eval 210 题套件 + 4 路评分 |

## 数据策略 (A+C+D)

### A. v3 DPO chosen 端 (~32k 主力)

来源: `/mnt/data/huangjiawei/datasets_local/medical_dpo/`

| 文件 | 量 | 来源模型 | 备注 |
|---|---|---|---|
| chosen_m2_18k_clean.jsonl | 12,798 | M2-32B | thinking 输出, 需检查长度 |
| chosen_opus_25k_clean.jsonl | 6,704 | Opus 4.7 | 警惕中英混杂 |
| chosen_taskd_m2_clean.jsonl | 19,691 | M2-32B | Task D 补漏 |
| chosen_rerun_bad_4307.jsonl | 4,307 | M2-32B | Task E rerun |
| **合计** | **43,500** | — | 去重前 |

**清洗要点**:
1. **去重**: prompt SHA-1 去重 (跨 4 个文件)
2. **过滤回避型**: 正则匹配 "建议咨询/请咨询/请就医/请到医院" 出现在回答开头 + 回答 < 200 字的样本删
3. **格式归一**: 统一为 `{"system": "...", "messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
4. **system prompt**: 单一 system, 不再做 v2 的 TCM/EBM 双轨 (双轨被证明带来 system→风格 强映射, 干扰诊断)
5. **chosen thinking 处理**: M2 thinking 段如果存在,选择性保留(看 smoke 测试结果)
6. **语种校验**: 抽 100 条断言 user/assistant 同语种, 中英混杂条丢弃

预计去重 + 过滤后 ~28-32k 干净样本。

### C. "不回避临床判断" 数据 (3-5k 关键)

**目标**: 治 v2 SFT 的硬伤 — open new_30 judge 4.20 跟 base 4.87 差 0.67 分, 来自 "让用户问医生" 这种回避型回答。

**生成方式**: 用 M2-32B in-process + 强 prompt (避开网关风控)

**Prompt 模板**:
```
你是一位资深临床医师。下面给你一个真实的患者咨询场景,请给出**完整、明确的临床判断**。

【硬规则】
1. 必须给出明确诊断/建议/用药/禁忌
2. 不允许说 "建议咨询医生" / "请到医院" 这种推卸
3. 必须列出具体药品名、剂量、疗程
4. 必须说明禁忌人群和警告

【场景】
{question}
```

**场景设计** (3-5k 题, 跟 v3 eval 互不重叠):
- 中医辨证 800 题 (病案 → 辨证 + 治法 + 方剂)
- 西医急诊 600 题 (典型 case + 处理流程)
- 用药咨询 800 题 (具体药品的使用 + 禁忌)
- 中西医结合 500 题 (病情 + 综合方案)
- 患者教育 500 题 (慢病管理 + 用药安全)
- 边界场景 800 题 (儿童/孕妇/老年/肝肾功能不全)

**生成预算**: 3-5k 题 × M2 ~30s/题 × 8 卡 work-stealing queue ≈ 1-2h DLC

**质检**: 5% 抽样人工核, 任何 "建议咨询" "请就医" 关键词出现 → 丢弃 + 报告

### D. 剂量类标准方剂 (1-2k 校准)

**目标**: 修 base/dpo 在评估中暴露的剂量错误 (六味地黄丸 / 肾上腺素剂量都错)

**来源**:
- 中医: 中国药典 2020 版方剂学 (300-500 个常用方)
- 西医: WHO Essential Medicines + 国家基本药物目录
- 急救: 急救医学指南 (肾上腺素 / 阿托品 / 硝酸甘油 等关键药)

**格式** (每条):
```
Q: {方剂名/药名} 的标准组成、剂量、主治、禁忌是?
A: {严格按照药典/指南填写,所有剂量数字必须可查}
```

**生成**: 不靠模型生成,**人工 + 工具脚本从公开数据库抠**。这部分量小但精度要求最高 — 它是 v4 比 v3 强的关键差异点。

**预算**: 1 周人工 + 脚本

## 训练配置

```bash
# 沿用 v2 配置, 改为纯 LLM
swift sft \
    --model /mnt/data/huangjiawei/models/Qwen3-8B \
    --train_type full \
    --tuner_type full \
    --dataset /mnt/data/huangjiawei/datasets_local/medical_v4/train_v4.jsonl \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --max_length 4096 \
    --gradient_accumulation_steps 4 \
    --batch_size 1 \
    --deepspeed zero3 \
    --gradient_checkpointing true \
    --save_steps 500 \
    --save_total_limit 5 \  # ⚠ 不再 2, 防止过拟合时无早期 ckpt 兜底 (v3 教训)
    --logging_steps 10 \
    --dataset_num_proc 8 \
    --eval_strategy no
```

ETA: 35k × 2 epoch / 16 卡 ≈ 8-12h

## 节奏: smoke → 全量

### Smoke (1k 子集, 30-60 min)

- 从 train_v4.jsonl 随机抽 1k (保证 A/C/D 比例)
- 单节点 8 卡 swift full SFT, 1 epoch
- 验证目标:
  1. **数据 schema 没坑** (chat_template 正确, 无字段缺失)
  2. **loss 下降** (step 100 时 loss 应明显低于初始)
  3. **memory(GiB) 稳定** (估 ~32-40 GB 单卡)
  4. **token_acc 上升** (>0.6 起步, 说明数据可学)

任一不满足 → 停止,排查数据/配置。

### 全量 (35-40k, 8-12h DLC)

smoke 通过后启动。配置同上,完整 2 epoch。

## 评估

直接复用 v3 eval 工程 (`reference_medical_v3_eval_workspace.md`):

```bash
# 推理
python3 infer.py --base /path/to/v4_ckpt --tag v4_sft --out outputs/v4_sft.jsonl

# 评分 (mcq + open + style)
python3 score.py --inputs outputs/v4_sft.jsonl ...
python3 style_metrics.py --inputs ...
python3 generate_report.py
```

**v4 必须达到的硬指标**:

| 维度 | v3 base | v3 dpo_1200 | v2 sft | **v4 目标** |
|---|---|---|---|---|
| CMB 医学 | 74% | 74% | 71% | ≥ 75% |
| C-Eval 通识 | 78% | 78% | 92% | ≥ 80% (不大幅遗忘) |
| open seed | 4.87 | 4.87 | 4.60 | ≥ 4.85 |
| open new_30 | 4.87 | 4.73 | **4.20** | **≥ 4.80** (修回避是关键) |
| 抽样剂量错误率 | ~30% | ~30% | ~10% | **< 5%** |

**任一不达标 → 失败,回到数据 prep 阶段调整 C/D 比例**。

## 风险点 (v4 可能踩的坑)

1. **A 数据 32k 清洗后剩多少未知** — 如果回避型 + 中英混杂太多, 可能只剩 20k, 需 C 补到 8-10k
2. **C 数据生成质量** — M2 跑 5k 题 + judge 抽审, 错留率 > 10% 就废
3. **D 数据来源**: 公开药典/指南可能没现成 API, 需要 PDF 抠 + 校对, 时间不可控
4. **single system prompt 是否足够** — v2 是双轨, 这次干脆单一(更干净), 但可能损失 system→风格映射, 通过 smoke 评估来看
5. **save_total_limit=5** — 防止 v3 的早期 ckpt 全没了的悲剧, 但占盘 5 × 17.5GB ≈ 87.5GB, 提前确认 sft_runs 盘空间

## 时间表 (估)

| 阶段 | 估时 |
|---|---|
| A 数据清洗 + 去重 | 0.5 天 |
| C 数据生成 (M2 跑) + 审核 | 1.5 天 |
| D 剂量数据整理 (人工 + 脚本) | 4-7 天 (主要瓶颈) |
| 训练集合并 + smoke | 0.5 天 |
| 全量训练 | 0.5 天 (DLC 8-12h) |
| 评估 + 报告 | 0.5 天 |
| **合计** | **~7-10 天** |

## 不做的事 (避免范围蔓延)

- ❌ 不做 RAG (终端目标是离线推理,RAG 是后续工程)
- ❌ 不做 v2 双轨 system 复刻
- ❌ 不做 LoRA (full SFT 显存够,不省时间)
- ❌ 不做 DPO (v3 教训, 同分布配对没解决前不开)
- ❌ 不做多轮对话训练 (单轮 instruction tuning 先,多轮后期再说)

## 待用户确认

1. ✅ 起点 Qwen3-8B base (已确认)
2. ✅ A+C+D 数据 (已确认)
3. ✅ 1k smoke 先跑 (已确认)
4. ⏸ **D 数据来源具体哪几本书/指南?** (用户医学背景方更清楚)
5. ⏸ **C 数据生成时机**: 现在跑还是等 A/D 完了再跑? (M2 占 8 卡 1-2h)
6. ⏸ **system prompt 具体写什么?** (单一版, 我可以起草让你过)

---

**下一步**: 等用户过完文档,开始 A 数据清洗(最快/最确定的一步)。
