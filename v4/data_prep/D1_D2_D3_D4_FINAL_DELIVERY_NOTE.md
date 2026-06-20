# D 数据检索交付说明（保守版）

生成日期：2026-06-19

## 交付文件

| 部分 | 主文件 | QA文件 | 数量 | 状态 |
|---|---|---|---:|---|
| D1 教材方剂 | `D1_300_formula_textbook_extracted.csv` | `D1_300_formula_textbook_QA.csv` | 300 | 来自用户提供的《方剂学》第十版 PDF 文本抽取；P0 119 条，P1 181 条 |
| D2 西药临床 | `D2_400_western_meds_conservative_merged.csv` | `D2_400_western_meds_conservative_merged_QA.csv` | 400 | 药名/规格底表来自卫健委《国家基本药物目录（2018年版）》PDF；P0 说明书字段合并早前 39 药品通预填，均需二核 |
| D3 急救药剂 | `D3_150_emergency_conservative.csv` | `D3_150_emergency_conservative_QA.csv` | 150 | P0 30 条急救剂量草稿；P1 120 条仅为检索占位，不填未经核验剂量 |
| D4 中成药禁忌 | `D4_60_supplement_filled_conservative.csv` | `D4_60_supplement_filled_conservative_QA.csv` | 60 | 早前完成；主要为非官方说明书预填，需 NMPA/原厂二核 |

## 关键边界

1. **不得直接入训**：D1/D2/D3/D4 均保留 QA 状态；凡 `需二核`、`待指南检索`、`注意/禁忌待补核`、`未定位`、`错配已清空` 的条目，必须人工核对原文后再转 jsonl。
2. **D1 剂量来源**：D1 的组成/剂量字段直接从 `C:\Users\PC\Downloads\08 方剂学第10版.pdf` 抽出的逐页文本生成，保留 `PDF页码`、`书内索引页码`、`原文片段文件`。
3. **D1 未定位项**：`人参养荣汤` 在该 PDF 中未按“方名→出处→【组成】”结构定位，已标记 `未定位`，不要入训。
4. **D2 来源限制**：`国家基本药物目录（2018年版）`只提供品种/剂型/规格，不提供完整适应症和剂量；剂量字段来自非官方 39 药品通预填时已标 `需NMPA/原厂二核`。
5. **D3 版本冲突**：用户清单中的成人心动过缓阿托品为 `0.5mg IV q3-5min，总量≤3mg`，当前 AHA 2020 成人流程常见为 `1mg IV q3-5min，最大3mg`；该条已标 `版本冲突待确认`，需确定采用指南版本。

## 摘要统计

### D1

- `D1_300_formula_textbook_extracted.csv`：300 条。
- P0：119 条；P1：181 条。
- QA：`已从教材PDF抽取待人工复核` 61 条，`注意/禁忌待补核` 235 条，`同名/多出处需复核` 3 条，`未定位` 1 条。
- 说明：中医教材很多方剂正文没有单独“禁忌”段，脚本只抽取正文中明确出现的禁忌/注意句，未抽到时不猜测。

### D2

- `D2_400_western_meds_conservative_merged.csv`：400 条。
- P0：80 条；P1：320 条。
- 成人剂量非空：67 条，均需 NMPA/CDE/原厂说明书或《临床用药须知》二核。
- QA：`需NMPA/原厂二核` 65 条，`字段不完整` 9 条，`未匹配说明书` 324 条，`错配已清空` 2 条。

### D3

- `D3_150_emergency_conservative.csv`：150 条。
- P0：30 条；P1：120 条。
- QA：`需PDF页码二核` 16 条，`版本冲突待确认` 1 条，`待指南检索` 13 条，`P1待指南检索` 120 条。
- P1 条目仅建立检索清单，剂量字段写为“待指南检索；未核验前不得入训”。

### D4

- `D4_60_supplement_filled_conservative.csv`：60 条。
- 来源多数为非官方说明书预填，已在 D4 交付说明中标注边界。

## 主要生成脚本

- D1 P0 抽取：`extract_d1_textbook.py`
- D1 300 扩展：`build_d1_300_from_textbook.py`
- D2 400 构建：`build_d2_400_from_neml_and_39ypk.py`
- D2 P0 合并：`merge_d2_p0_into_400.py`
- D3 150 构建：`build_d3_150_emergency.py`

## 原始/中间材料

- D1 PDF 逐页文本：`fangjixue10_pages_text/`
- D1 P0 原文片段：`D1_formula_snippets/`
- D1 300 P1 原文片段：`D1_formula_snippets_300/`
- D2 官方目录 PDF：`NHC_NEML_2018.pdf`
- D2 官方目录逐页文本：`NHC_NEML_2018_pages_text/`
- D2 说明书网页缓存：`D2_400_39ypk_raw/`、`D2_39ypk_raw/`

## 建议下一步

1. 先人工核对 D1 的 P0 119 条剂量与页码，尤其 `六味地黄丸`、`肾气丸`、附方条目和 `同名/多出处需复核` 条目。
2. 对 D2 的 80 条 P0，优先用 NMPA/CDE/原厂说明书或《临床用药须知》替换非官方说明书来源。
3. 对 D3 的 30 条 P0，逐条下载指南 PDF 并补 `页码/章节/原文片段`；心动过缓阿托品剂量需先确定指南版本。
4. 仅把 QA 通过且原文可追溯的行转成 jsonl 入训。
