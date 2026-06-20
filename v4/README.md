# v4: Qwen3-8B Full SFT(2026-06-20 完工)

> v3 DPO 跨模型族空转 → v2 SFT 回避临床判断 → 这次 v4 是**第一个真正可交付的 ckpt**。

## 配套 HuggingFace

- **模型**: [shdkahjkda/medical-sft-v4-qwen3-8b](https://huggingface.co/shdkahjkda/medical-sft-v4-qwen3-8b)
- **数据**: [shdkahjkda/medical-sft-v4-data](https://huggingface.co/datasets/shdkahjkda/medical-sft-v4-data)

## 评估结果(1390 题套件,跟 base/v2/dpo/M2 对照)

| 指标 | qwen3-8b base | v2 SFT | v3 DPO | **v4 SFT** | M2-32B |
|---|---|---|---|---|---|
| CMB 235 | 72.3% | 71% | 72.3% | **69.8%** | 47.2% |
| MedQA-CN 1000 | 80.8% | 80.1% | 81.1% | **80.1%** | 47.1% |
| C-Eval 50 | 80% | 92% | 80% | **84%** | 58% |
| open_v2_seed | 4.73 | 4.60 | 4.73 | **5.0** | 5.0 |
| **open_new_30** | 4.73 | **4.20** | 4.73 | **5.0** | 5.0 |

## 工程文件

```
v4/
├── README.md                 # 这个文件
├── PLAN.md                   # 项目规划(2026-06-19 起草)
├── HF_DATASET_README.md
├── HF_MODEL_README.md
├── data_prep/                # 数据 prep 脚本
│   ├── A_01_report.py        # A 数据统计报告
│   ├── A_02_filter.py        # A 数据过滤
│   ├── D1_to_jsonl.py        # 教材方剂 csv → SFT jsonl
│   └── ... (D 系列、smoke 抽样、QA 等)
├── scripts/                  # 训练 + 数据生成
│   ├── C_gen_m2.py           # C 单节点版
│   ├── C_gen_m2_4node.py     # C 4 节点 work-stealing 版(实跑)
│   ├── C_gen_m2_4node.sh     # C DLC launcher
│   ├── C_smoke_50.py         # C max_tokens smoke
│   └── merge_v4_train.py     # 合并 C→D1→A 切 thinking + 统一 schema
└── eval_v2/                  # 1390 题评估
    ├── download_medqa.py
    ├── assemble_eval_v2.py   # CMB 280 + MedQA 1000 + C-Eval 50 + 60 open
    └── run_eval_v2.sh        # 4 模型推理 launcher
```

## 数据策略

37,353 条 = A 主力 + C 修硬伤 + D1 修剂量

| 子集 | 量 | 来源 | 解决问题 |
|---|---|---|---|
| **A** v3 chosen 过滤 | 32,078 | M2-32B + Opus 4.7 long-form | medical advice 主体 |
| **C** M2 重写 evasive | 4,979 | M2 强 system prompt 重生成 | 修 v2 "请咨询医生" 回避硬伤 |
| **D1** 教材方剂 | 296 | 《方剂学》第十版 PDF 抽取 | 修 base/dpo "六味地黄丸 9g/9g/..." 剂量错 |

## 训练配置

```bash
# DLC 4 节点 × 8 GPU = 32 卡
# 1h 14m 跑完 584 step
swift sft \
  --model Qwen/Qwen3-8B \
  --tuner_type full \
  --dataset v4_train_full.jsonl \
  --num_train_epochs 2 \
  --learning_rate 5e-6 \
  --max_length 4096 \
  --gradient_accumulation_steps 4 \
  --deepspeed zero3 \
  --gradient_checkpointing true \
  --save_steps 200 \
  --save_total_limit 10  # 防止早期 ckpt 被覆盖
```

## 关键教训(写进了 memory)

1. **DPO 跨模型族 chosen/rejected 必空转**(`feedback_dpo_cross_family_zero.md`)
   - v3 DPO 80 GPU-h 烧零提升 → 切回 SFT
2. **SFT 训练集合并必须强制统一 schema**(`feedback_sft_merge_unified_schema.md`)
   - 多源 jsonl 字段不一致 → HF datasets DatasetGenerationError
3. **long-form SFT 客观题 -5% 是小样本噪声**(`feedback_long_form_sft_mcq_tradeoff.md`)
   - 100 题 -5% / 1390 题 -2.5%(置信区间内),不是必然趋势
4. **DLC 控制台空白 ≠ 任务挂**(`feedback_dlc_console_tee_buffer.md`)
   - `| tee` 块缓冲,用 `stdbuf -oL -eL tee` 修

## License

- 基模 Qwen3-8B: Apache 2.0
- 数据二次合成(Opus 蒸馏部分): **Anthropic ToS 商用风险**
- D1 教材抽取: 仅供研究使用,商用前需取得人卫出版社授权
