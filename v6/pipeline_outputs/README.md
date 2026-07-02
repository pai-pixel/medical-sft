# pipeline_outputs — v6 造数据流水线的中间产物

留档一份 v6 造数据流水线各阶段的中间产物,便于复现 / 审计 / 抽样分析。**最终合成数据(v6_train_full 56k)在 HF `shdkahjkda/medical-sft-v6-data`,不在本目录**。

## 目录结构

```
pipeline_outputs/
├── seed_prompts/              Phase 1 + Phase 2 seed prompts 各源产物
│   ├── hf_cmb_train_1000.jsonl              (390K)  HF CMB 抽 1000 真 MCQ 带 GT
│   ├── hf_medical_o1_acute_3000.jsonl       (1.2M)  HF medical-o1-reasoning 过滤儿科/急救 3000
│   ├── hf_short_qa_4000.jsonl               (3.3M)  HF ChatDoctor-HealthCareMagic-100k-CN 抽 4000
│   ├── d1_textbook_tcm_seed.jsonl           (265K)  v4 D1 教材 296 → 1000 TCM seed (5 变体)
│   ├── opus_mcq_gen_gen.jsonl               (2.3M)  Opus 造 MCQ 4000
│   ├── opus_acute_gen.jsonl                 (833K)  Opus 造急症 1998
│   ├── opus_tcm_gen.jsonl                   (1.5M)  Opus 造 TCM 4000
│   └── opus_short_gen.jsonl                 (236K)  Opus 造短问 1000
├── merged_prompts.jsonl       (9.5M)  Phase 2b 合并 + 去重 19710(送给 opus_gen_answers 的输入)
├── merge_stats.json           (560B)  合并分布统计
└── answers/
    └── opus_all_raw.jsonl     (74M)   Phase 3 raw output 19660 条(filter 前)
```

## 用途

- **复现**: `scripts/filter_answers.py` 消费 `answers/opus_all_raw.jsonl` → `v6_new_clean.jsonl (19555)`
- **审计**: `answers/opus_all_raw.jsonl` 保留了 filter 前的 105 条被剔除样本(细节 filter 规则命中)
- **抽样**: 想看单 category / 单 teacher / 单 wave 的分布,从这些中间产物比从最终合并版更方便
- **调整**: 若日后要改 filter 规则,不用重跑 Opus,直接消费 `answers/opus_all_raw.jsonl`

## Schema

### seed_prompts/*.jsonl
```json
{"id": "...", "prompt": "...", "source": "...", "category": "mcq|acute|tcm|short", "gt_answer": "A|B|C|D|null", "meta": {...}}
```

### merged_prompts.jsonl
同上 schema,已按 md5 hash 去重,category 已 normalize(mcq_gen → mcq)。

### answers/opus_all_raw.jsonl (Phase 3 output)
```json
{
  "id": "...",
  "prompt": "...",
  "category": "mcq|acute|tcm|short",
  "source": "...",
  "gt_answer": "A|B|C|D|null",
  "answer": "<think>...</think>\n<final>",
  "finish_reason": "stop|length",
  "teacher": "opus-4-7 | opus-4-6-ksyun"
}
```

## Teacher 混合

`answers/opus_all_raw.jsonl` 里 3 种 teacher tag:
- **opus-4-7** (最初的 tag,首次跑 2797 条)—— `claude-opus-4-7:mindracode-anthropic-qianli`
- **opus-4-6-ksyun** (第二段 1095 条)—— `claude-opus-4-6:ksyun-aws`(切换期,网关抖动)
- **opus-4-7** (第三段主力 15770+ 条)—— 恢复后主路

按 teacher grep 可分层评估。

## 关联

- 数据集: [`shdkahjkda/medical-sft-v6-data`](https://huggingface.co/datasets/shdkahjkda/medical-sft-v6-data)
- 消费脚本: 见上级目录 `scripts/filter_answers.py` / `scripts/merge_v6_train.py`
