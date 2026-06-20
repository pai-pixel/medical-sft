# D 数据项目记忆

更新时间：2026-06-20

## 项目定位

本项目位于 `C:\Users\PC\Claude脚本\medical_sft\v4\data_prep`，目标是为 medical SFT/DPO 的 D 类剂量数据构建可追溯检索表，重点修复 base/dpo 在剂量题中暴露的错误，尤其是：

- 中医方剂剂量，例如 `六味地黄丸`。
- 急救剂量，例如 `肾上腺素`。

核心原则：**剂量数字宁可留空/标 QA，也不能编造或未核验入训**。

## 当前交付状态

主交付文件：

- D1 方剂 300 条：`D1_300_formula_textbook_extracted.csv`
- D1 QA：`D1_300_formula_textbook_QA.csv`
- D2 西药 400 条：`D2_400_western_meds_conservative_merged.csv`
- D2 QA：`D2_400_western_meds_conservative_merged_QA.csv`
- D3 急救 150 条：`D3_150_emergency_conservative.csv`
- D3 QA：`D3_150_emergency_conservative_QA.csv`
- D4 中成药 60 条：`D4_60_supplement_filled_conservative.csv`
- D4 QA：`D4_60_supplement_filled_conservative_QA.csv`
- 总说明：`D1_D2_D3_D4_FINAL_DELIVERY_NOTE.md`

数量已核验：

- D1：300 条。
- D2：400 条。
- D3：150 条。
- D4：60 条。

## D1 方剂经验

用户提供 PDF：

- `C:\Users\PC\Downloads\08 方剂学第10版.pdf`

已抽取逐页文本目录：

- `fangjixue10_pages_text/page_001.txt` 到 `page_559.txt`

关键脚本：

- `extract_d1_textbook.py`：抽 D1 P0 119 条。
- `build_d1_300_from_textbook.py`：从书后方名索引扩展到 300 条。

重要结论：

- 《方剂学》第十版 PDF 实际按 21 章组织，不只是用户最初说的 17 章，包含 `表里双解剂`、`涌吐剂`、`治痈疡剂` 等。
- D1 P0 最终按用户清单是 119 条，不是严格 100 条。
- `人参养荣汤` 未能在该 PDF 正文按“方名→出处→【组成】”结构定位，只出现过类似 `人参养荣丸` 医案，已标 `未定位`，不要入训。
- `六味地黄丸` 已定位：PDF P.258，书内索引 P.137；组成保留原文剂量：`熟地黄炒，八钱（24g） 山萸肉 干山药各四钱（各12g） 泽泻 牡丹皮 茯苓去皮，各三钱（各9g）`。
- 教材中很多方剂没有独立“禁忌/注意”段，脚本只抽正文中明确出现的禁忌/注意句，没抽到就标 `注意/禁忌待补核`，不要猜。
- 附方解析一开始会把用法句污染进组成字段，已在 `extract_d1_textbook.py` 中修正为遇到 `上/以水/水煎服/每服/先煮` 等就切分。

## D2 西药经验

官方底表 PDF：

- `NHC_NEML_2018.pdf`
- 下载自卫健委官方《国家基本药物目录（2018年版）》PDF。

关键脚本：

- `build_d2_400_from_neml_and_39ypk.py`
- `merge_d2_p0_into_400.py`
- 早前 P0 抓取：`fill_d2_from_39ypk.py`
- 早前 P0 保守清洗：`finalize_d2_conservative.py`

重要边界：

- 《国家基本药物目录（2018年版）》只能作为品种/剂型/规格底表，不能提供完整适应症、成人剂量、禁忌、不良反应。
- D2 P0 的部分说明书字段来自 39 药品通，必须标 `需NMPA/原厂二核`，不可直接训练。
- D2 合并版保留了早前 80 条 P0 预填成果；其中成人剂量非空 67 条。
- `吗啡` 早前错配为检测试纸，已清空并标 `错配已清空`。
- `口服补液盐III` 早前错配为口服补液盐散Ⅱ，已清空并标 `错配已清空`。
- `肾上腺素` 不在 D2 P0 清单里，急救剂量在 D3 处理。

踩坑：

- NMPA 数据站可访问，但 API 需要 `timestamp` 和 `sign`，前端 JS 有 `appSecret = nmpasecret2020`，签名逻辑尚未完全复现，之前尝试返回 `请求签名验证失败`。
- CDE 页面存在反爬/202 保护，未能批量稳定抓标签。
- 39 药品通只是非官方预填源，不能作为最终训练出处。
- PDF 解析国家目录时，药名/英文/规格跨行会混乱，最后以“够 400 条底表 + QA 标记”的保守方式交付，不强行声称全解析完美。

## D3 急救经验

关键脚本：

- `build_d3_150_emergency.py`
- 初始草稿：`D3_P0_30_emergency_draft.csv`

重要边界：

- P0 30 条保留常用剂量草稿，但均需指南 PDF 页码二核。
- P1 120 条只做检索占位，剂量字段写成 `待指南检索；未核验前不得入训`，避免污染。
- D3 全表不可直接入训，必须用 AHA/中国指南/药品说明书逐条补 PDF 页码和章节。

关键冲突：

- 用户最初清单写成人心动过缓：`阿托品 0.5mg IV q3-5min，总量≤3mg`。
- 2020 AHA 成人心动过缓流程常见为：`阿托品 1mg IV q3-5min，最大3mg`。
- 当前 D3 已将心动过缓标为 `版本冲突待确认`，下次必须先确认采用哪版指南。

重点剂量样例：

- 过敏性休克：肾上腺素成人 IM 0.3–0.5mg；儿童 0.01mg/kg；需 WAO/中国急诊指南页码二核。
- 心脏骤停：肾上腺素成人 IV/IO 1mg q3–5min；需 AHA 2020 页码二核。
- 室颤/无脉室速：胺碘酮 300mg 首次，150mg 第二次；需 AHA 2020 页码二核。
- 急性缺血性脑卒中溶栓：阿替普酶 0.9mg/kg，最大 90mg，10% 1min bolus，余 90% 60min；需卒中指南/说明书页码二核。

## D4 中成药经验

用户已给模板：

- `D4_60_supplement_template.csv`

交付文件：

- `D4_60_supplement_filled_conservative.csv`
- `D4_60_supplement_filled_conservative_QA.csv`
- `D4_60_supplement_filled_conservative_nonofficial_usable.csv`
- `D4_DELIVERY_NOTE.md`

边界：

- D4 多数来自 39 药品通说明书预填，非 NMPA/原厂官方，必须二核。
- D4 是补 `[禁忌]`、`[不良反应]`、`[注意事项]`，不要改用户已有的药典处方/功能主治/用法用量。

## Windows/编码坑

PowerShell + 中文常见坑：

- PowerShell here-string 中写中文 Python 字面量，有时会变成 `????` 或键名乱码。
- 读取 UTF-8 文件时，PowerShell `Get-Content` 显示可能乱码，但文件本身是 UTF-8 正常。
- 建议每次运行前设置：

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING='utf-8'
```

仍可能有中文字面量问题时：

- Python 脚本里用 Unicode escape，例如 `'\u6842\u679d\u6c64'`。
- CSV 用 `csv.reader` 按列号读，避免中文字段名在 here-string 中被污染。
- 文件名尽量用 ASCII，例如 `NHC_NEML_2018.pdf`，不要直接写中文 PDF 文件名，曾遇到 `OSError: Invalid argument: '????????.pdf'`。

## 推荐续做顺序

1. **D1 人工核 P0 119 条**：先核剂量、PDF页码、书内索引页码；`人参养荣汤` 不入训。
2. **D2 P0 80 条官方化**：优先从 NMPA/CDE/原厂 PDF/《临床用药须知》替换 39 药品通字段。
3. **D3 P0 30 条页码化**：下载指南 PDF，补 `页码/章节/原文片段`；先解决阿托品 0.5mg vs 1mg 版本问题。
4. **D4 60 条官方化**：用 NMPA/原厂说明书补齐禁忌、不良反应、注意事项。
5. 仅将 QA 通过、剂量原文可追溯的行转 jsonl 入训。

## 工作准则

- 不要为了凑数而填未经核验剂量。
- 不要把非官方说明书标成官方。
- 不要把“注意/禁忌待补核”的中医方剂直接转训练。
- 不要让急救 P1 占位行进入训练集。
- 任何剂量数字必须能回到具体 PDF 页码、指南章节或说明书原文。
