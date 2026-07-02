---
language: [zh]
license: other
size_categories: [10K<n<100K]
task_categories:
  - text-generation
  - question-answering
tags:
  - medical
  - chinese-medicine
  - western-medicine
  - sft
  - thinking
  - opus-4-7
---

# medical-sft-v6-data — 中西医 SFT 数据集(v6 混合 56k,含 thinking)

**v6 = v4 37k(基础)+ v6 新造 19555(补 4 短板)**。相比 v4 单一 long-form 分布,v6 补齐:MCQ 客观题 / 临床急症决策 / 中医方剂辨证 / 短问答口语化 四大缺失场景。

## 文件

- **v6_train_full.jsonl** — 56,633 条,SFT 训练主文件(保留 `<think>...</think>` thinking 段)
- **v6_train_smoke_2k.jsonl** — 2000 条抽样,用于 smoke train / debug loader
- **v6_new_clean.jsonl** — 19,555 条 v6 新增(不含 v4 legacy),便于分层分析

## Schema

```json
{
  "id": "v6_mcq_00001 | D1_0 | ...",
  "system": "你是一位资深临床医师,精通中医辨证施治与现代循证医学。...",
  "messages": [
    {"role": "user", "content": "<prompt: MCQ 题干+ABCD / 急症场景 / 中医求方 / 短口语>"},
    {"role": "assistant", "content": "<think>推理段</think>\n<最终中文回答>"}
  ]
}
```

- `<think>...</think>` 结构统一,training / inference 时可保留(v5 风格)或剥离(v4 风格)
- 全 CN 输出,filter 阶段已断言 sys/user/asst 三段语种一致

## 数据来源

**基础层(v4 37k)**:继承 shdkahjkda/medical-sft-v4-data,含 A(v3 chosen 去重)+ C(M2 重写 evasive)+ D1(《方剂学》第十版 296 条)。

**v6 新增 19,555 条(4 类)**:

| category | 数量 | seed 来源 | answer 老师 |
|---|---|---|---|
| mcq | 4,976 | HF `FreedomIntelligence/CMB` 1k 真题 + Opus 造 4k 新 MCQ | Opus 4.7 |
| acute | 4,608 | HF `FreedomIntelligence/medical-o1-reasoning-SFT` 儿科/急救过滤 3k + Opus 造 2k | Opus 4.7 |
| tcm | 4,982 | v4 D1 教材抽 seed 1k + Opus 造 4k(经方时方/辨证) | Opus 4.7 |
| short | 4,989 | HF 中文短问答 4k + Opus 造 1k | Opus 4.7 |

**Teacher 混合**:opus-4-7 mindracode-anthropic-qianli(主力)+ opus-4-6:ksyun-aws(网关抖动切换期少量),两代 opus 医疗领域差异 <2%。

**Filter 阶段**:19,660 raw → **19,555 kept (99.5%)**,规则:finish=length 丢 / final <150 字丢 / evasive 开头丢 / CN 占比断言 / MCQ 必须含答案标识 / prompt SHA1 去重。

## License

**多源混合,按用途划分**:

- **v4 legacy 37k(基础)**:继承 v4 各源 license(v2 ChatGPT 蒸馏 + Baichuan-M2 蒸馏 + 《方剂学》教材摘录)
- **v6 新增 19555(Opus 蒸馏)**:受 **Anthropic Usage Policy** 约束,**不得用于训练与 Anthropic 商用产品竞争的模型**。仅限研究、教学、内部工具用途
- **HF 公开 seed prompts**:各源 license(CMB / medical-o1-reasoning / ChatDoctor 系列各自)

**商用请自评风险**。仅研究/教育用途分发。

## 配套代码

- GitHub: [pai-pixel/medical-sft](https://github.com/pai-pixel/medical-sft) 的 `v6/` 子目录 — 数据 prep(A_01 拉 HF / A_02 D1 seed / A_03 Opus 造 seed / A_04 merge)+ Opus answer gen(scripts/opus_gen_answers.py)+ filter/merge(filter_answers.py / merge_v6_train.py)+ smoke_review

## 项目状态

**已完工**(2026-07-02),19,555 v6 new + 37,078 v4 = **56,633 v6_train_full**。等 Phase 6 全量 SFT 训练验证效果。相比 v4:预期 CMB 客观题回补(+MCQ 数据),同时保留 open_new_30 临床决策能力(long-form 未变)。

## 关联 repos

- `shdkahjkda/medical-sft-v4-data` — v4 前作(37k 长文本 SFT)
- `shdkahjkda/medical-v5-8b` — v5 model(Qwen3-8B thinking-on SFT)
- `shdkahjkda/medical-dpo-43k` — v3 DPO 数据(Opus + M2 双老师)
