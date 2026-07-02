# v6 — 中西医 SFT 数据集扩充(20k+ 混合 QA)

> 在 v4 37k 基础上,由 Opus 4.7 蒸馏 19,555 条,补齐 4 类短板:MCQ 客观题 / 临床急症决策 / 中医方剂辨证 / 短问答口语化。合并后 v6_train_full = **56,633 条**。

**当前状态**(2026-07-02):数据产出完工,已推 HF,等 Phase 6 全量 SFT 训练。

## 立项动机

v4/v5 已交付,但两个短板依然明显:
- **v4 CMB -5%**: 训练数据 87% 是 long-form,MCQ 客观题能力被稀释(`docs/lessons-learned.md`)
- **分布单一**: 37k 数据几乎全是 v2 chosen + M2 重写 + 教材,缺真实用户口吻 / 急症场景 / 短问答

## 数据分层(19,555 v6 新增)

| category | 数量 | seed 来源 | answer 老师 |
|---|---|---|---|
| **mcq** | 4,976 | HF `FreedomIntelligence/CMB` 1k 真题(带 GT)+ Opus 造 4k 新 MCQ | Opus 4.7 |
| **acute** | 4,608 | HF `medical-o1-reasoning-SFT` 儿科/急救/孕妇过滤 3k + Opus 造 2k | Opus 4.7 |
| **tcm** | 4,982 | v4 D1 教材 296 条抽 seed 1k + Opus 造 4k(经方/时方/辨证) | Opus 4.7 |
| **short** | 4,989 | HF `ChatDoctor-HealthCareMagic-100k-CN` 抽 4k + Opus 造 1k | Opus 4.7 |

## 关键约束(已落地)

- **thinking 段保留**:每条 answer 是 `<think>...</think>\n<final>`,格式跟 v5 一致
- **全 CN 硬约束**:SYS COMMON_RULES 强制"包括 thinking 全段中文";filter 阶段再 assert CN 占比
- **格式规整 vs 规则过滤**双保险:SYS 引导降低 4% → <1%,filter 兜底剔漏网
- **MCQ 客观题 GT 保真**:CMB train 抽的 1k 用题库自带 GT,Opus 只写 explanation
- **Teacher 混合**:opus-4-7 主力 + opus-4-6:ksyun-aws 少量(网关抖动切换期),两代 opus 医疗差 <2%

## 工程流水(实施顺序)

```
Phase 1  HF+D1 seed 收集(9,000)
Phase 2  Opus 造补 seed(10,998)
Phase 2b merge_prompts → 19,710 unique + smoke 100 验 pipeline
Phase 3  Opus 4.7 造 20k answers (SEM=50, 7h5m) → 19,660 落盘
Phase 4  filter (99.5% pass) → 19,555 clean + merge v4 37k → 56,633
Phase 5  推 HF + GitHub(本次)
Phase 6  v6 全量 SFT 4-node DLC(待启)
```

## 文件结构

```
v6/
├── README.md                        本文件
├── README_HF.md                     HF README 副本(推 HF 用)
├── data_prep/
│   ├── A_01_pull_hf_prompts.py      拉 HF 3 dataset(o1_acute / short_qa / CMB fallback)
│   ├── A_01b_pull_cmb_direct.py     单拉 CMB(绕开 datasets library schema 校验)
│   ├── A_02_extract_d1_tcm_seed.py  v4 D1 296 教材 → 1000 TCM seed(5 变体)
│   ├── A_03_opus_gen_prompts.py     Opus 造补 10,998 seed(4 类)
│   ├── A_04_merge_prompts.py        合并 + 去重 + category 打标 → 19,710
│   └── A_05_smoke_100.py            抽样 100 条验证
├── scripts/
│   ├── opus_gen_answers.py          ★ Phase 3 主戏:Opus 造 answer with thinking
│   ├── filter_answers.py            evasive / 短 / CN 断言 过滤
│   ├── merge_v6_train.py            v4 37k + v6 19555 → 56,633
│   ├── check_truncation.py          finish_reason 分位数分析(SOP 铁律)
│   ├── smoke_review.py              抽样审 thinking / 语种 / 长度
│   └── smoke_lang_check.py          英文短语精准检测
```

## 关键坑防守(全踩过)

| 坑 | 现象 | 修法 |
|---|---|---|
| **CMB dataset schema 不一致** | `load_dataset` train 缺 id / test 缺 answer 挂 | 绕开 datasets library,`hf_hub_download` 手 parse JSON |
| **A_03 chunk 太大** | WAVE=100 → 17 min 才 print 一次,用户看死 | WAVE=30 每 5 call 打进度 |
| **max_tokens 撞顶** | 未定 → 部分截断污染数据 | 先 smoke 看 p99 分布,现设 5120 覆盖 tcm max 3860 tokens (75%) |
| **thinking 英文推理** | acute 类 Opus 用 English 推理 | SYS COMMON_RULES 硬约束"含 thinking 全中文" |
| **anthropic 官方分组下线** | Wave 2600 突然 503 → abort | 切 opus-4-6:ksyun-aws |
| **ksyun-aws temp/top_p 冲突** | 424 error(网关代理自动补 top_p) | 全改 top_p(0.7 MCQ / 0.95 其他),不传 temperature |
| **virtiofs 读大文件慢** | Python 直接 open jsonl 卡死 | 先 `cp /tmp` 再读 |

## HF 数据

- [shdkahjkda/medical-sft-v6-data](https://huggingface.co/datasets/shdkahjkda/medical-sft-v6-data)(private)
  - `v6_train_full.jsonl` — 56,633 条(v6+v4 合并)
  - `v6_new_clean.jsonl` — 19,555 条(v6 新增,便于分层分析)
  - `v6_train_smoke_2k.jsonl` — 2000 抽样

## 用法(拉数据训 v6)

```python
from datasets import load_dataset
ds = load_dataset("shdkahjkda/medical-sft-v6-data", split="train")  # 56633 条

# 每条 schema: {id, system, messages: [user, assistant]}
# assistant 含 <think>...</think> + final
```

## 关联

- 前作 `v4/` — 37k 基础 SFT
- 前作 `v5/`(未在本仓,独立 repo `pai-pixel/medical-v5`)— thinking-on 版
- HF v4 数据 [`shdkahjkda/medical-sft-v4-data`](https://huggingface.co/datasets/shdkahjkda/medical-sft-v4-data)
