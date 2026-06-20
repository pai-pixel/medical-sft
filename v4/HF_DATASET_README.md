---
language:
- zh
license: other
size_categories:
- 10K<n<100K
task_categories:
- text-generation
- question-answering
tags:
- medical
- chinese-medicine
- tcm
- western-medicine
- sft
- v4
---

# medical-sft-v4-data — Qwen3-8B 中西医 SFT 训练集 + 1390 题评估集

v4 SFT(2026-06-20 完工)的训练数据 + 评估题集。配 v3 SFT(`medical-dpo-43k`)、v2 SFT(`medical-sft-v2`)对照看,这次 v4 是首个**真正可交付的 ckpt**:修住了 v2 "回避临床判断" 硬伤,教材方剂剂量正确,客观题持平 base。

## 文件

### `train/`(SFT 训练数据)

| 文件 | 行数 | 大小 | 描述 |
|---|---|---|---|
| `v4_train_full.jsonl` | 37,353 | 228 MB | **最终训练集**,A 32k + C 5k + D1 296,统一 schema (id+system+messages) |
| `v4_C_chosen_raw.jsonl` | 5,000 | 48.5 MB | C 原始 M2 输出(含 thinking),smoke max_tokens=4608 验证 0 截断 |
| `v4_D1_chosen.jsonl` | 296 | 1 MB | D1 教材方剂(《方剂学》第十版 PDF 抽取) |

### `eval/`(评估题集)

| 文件 | 行数 | 描述 |
|---|---|---|
| `eval_v2_1390.jsonl` | 1,345 | v2_seed 30 + new_30 30 + CMB 235 + MedQA-CN 1000 + C-Eval 50 |

## Schema

### 训练集
```json
{
  "id": "v4_C_00001",
  "system": "你是一位资深临床医师,精通中医辨证施治与现代循证医学...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
**严格 3 字段,丢一切元数据**(吸取多源合并 schema 不一致踩坑教训)。

### 评估集
```json
{
  "id": "cmb_001 / medqa_0001 / seed_S001 / new_N016 / ceval_001",
  "source": "CMB / MedQA / v2_seed / new_30 / C-Eval",
  "type": "open / mcq",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},  // mcq 才有
  "answer": "B",  // mcq gold letter
  "reference_answer": "...",  // open 才有
  "key_points": [...]  // open 才有
}
```

## 数据来源

### A 子集(32k 主力)
- 来源: v3 DPO chosen 端去重过滤(`medical-dpo-43k` 的 chosen 部分)
- 老师模型: Baichuan-M2-32B + Anthropic Claude Opus 4.7
- 用途: medical advice long-form 风格

### C 子集(5k,修 v2 回避硬伤)
- 来源: A 中含 evasive 标志的 prompt → M2 强 system prompt 重新生成
- 关键修复: 严格禁止"建议咨询医生"等回避语,必须给完整临床判断
- max_tokens 4608,smoke 实测 0 撞顶(p99=3828)
- thinking 段已切除统一为 final answer

### D1 子集(296 教材方剂)
- 来源: 《方剂学》第十版 PDF(邓中甲主编,人民卫生出版社 2016,ISBN 9787117218900)
- 抽取脚本验证: 桂枝汤/六味地黄丸/麻黄汤/理中丸 P0 4 题剂量 100% 正确
- v3 评估时 base/dpo/v2 全错的 "六味地黄丸 8:4:4:3:3:3" 比例,v4 学到了

### 评估题集
- CMB(FreedomIntelligence): val 280 中筛单选 235
- MedQA-CN(bigbio/med_qa): test 3426 中抽 1000
- C-Eval: 16 个非医疗学科共 50 题(灾难遗忘对照)
- v2_seed 30 + new_30 30 = 60 道开放问答(judge 评分)

## License

| 子集 | License |
|---|---|
| A 子集(M2/Opus 蒸馏) | Apache 2.0(M2 base) + **Anthropic ToS 商用风险**(Opus 部分) |
| C 子集(M2 蒸馏) | Apache 2.0 |
| D1 子集(教材抽取) | 仅供研究使用,商用前需取得人卫出版社授权 |
| 评估集 CMB | FreedomIntelligence 原 license |
| 评估集 MedQA | bigbio/med_qa 原 license |

## 训练效果

跟 v3 base / v3 dpo / v2 / M2-32B 在 1390 题套件对照:

| 指标 | v3_base | v3_dpo_1200 | v2_sft | **v4_sft** | M2-32B |
|---|---|---|---|---|---|
| CMB 235 | 72.3% | 72.3% | 71% | **69.8%** | 47.2% |
| MedQA 1000 | 80.8% | 81.1% | 80.1% | **80.1%** | 47.1% |
| C-Eval 50 | 80% | 80% | 92% | **84%** | 58% |
| open_v2_seed | 4.73 | 4.73 | 4.60 | **5.0** | 5.0 |
| **open_new_30(临床决策)** | 4.73 | 4.73 | **4.20** | **5.0** | 5.0 |

v4 在 open judge 跟 M2-32B 同档 5.0(临床咨询能力 32B → 8B 蒸馏成功),**修住 v2 4.20 → 5.0 硬伤**。

## 配套代码

GitHub: https://github.com/pai-pixel/medical-sft (`v4/` 子目录)

## 项目状态

- ✅ v4 SFT 完工,**唯一可交付 ckpt**
- ⏸ v5 含 thinking 重训进行中(2026-06-20)
- 配套模型: [shdkahjkda/medical-sft-v4-qwen3-8b](https://huggingface.co/shdkahjkda/medical-sft-v4-qwen3-8b)
