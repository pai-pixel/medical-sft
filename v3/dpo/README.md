# v3 DPO — 双老师蒸馏

> v3 第二阶段:在 GRPO 闭环验证后,转向 **DPO 偏好学习**,把强老师 (Baichuan-M2-32B 中医 + Claude Opus 4.7 西医) 的能力蒸馏到 Qwen3-8B-Instruct.

## 学生 + 老师 + 数据

```
学生   : Qwen3-8B-Instruct  (LoRA r=16, 训出来部署到 RK3588 32GB 边缘终端)
老师 chosen:
  ├─ Baichuan-M2-32B  (TCM 中医辨证 / 方剂 / 安全引导, 7 题测试 7/7 全过)
  └─ Opus 4.7         (EBM 西医循证 / 红旗症状 / 急救分诊)
学生 rejected: Qwen3-8B-Instruct 自身生成 (Task B, 同 base)
prompt 池: 43,500 条 (TCM 18002 + EBM 17998 + General 7500)
```

按 domain 路由: TCM → M2 老师, EBM/General → Opus 老师, 错配域由 M2 兜底.

## Phase 2 数据生成 (4 件套 chosen + 1 件 rejected, 共 43,500 配对)

| 任务 | 文件 (本地, 已 .gitignore) | 模型 | clean (token 法过滤后) | bad (撞 max_tokens) |
|---|---|---|---|---|
| Task A | chosen_m2_18k_clean.jsonl | M2-32B vllm | 12,798 | 747 |
| Task B | rejected_qwen3_43k.jsonl | Qwen3-8B 自答 | 43,500 | (无需过滤) |
| Task C | chosen_opus_25k_clean.jsonl | Opus 网关 | 6,704 | 0 |
| Task D | chosen_taskd_m2_clean.jsonl | M2-32B 兜底 | 19,691 | 3,560 |
| Task E | chosen_rerun_bad_4307.jsonl | M2-32B 重跑撞顶 | 4,307 | 12 |
| **合计** | dpo_train_43k.jsonl | | **43,488 配对** | |

## 文件清单

### 数据生成 / 路由
- `01_sample_v2_schema.py` / `02_sample_v2_prompts.py` — v2 数据集 user 段抽样
- `03_smoke_quality_filter.py` / `04_full_quality_filter.py` — Sonnet 4.6 质量过滤
- `05_smoke_opus_prompt_gen.py` / `06_full_opus_prompt_gen.py` — Opus 自生成 prompt (safety/refuse 类)
- `07_merge_prompt_pool.py` — 43.5k prompt 池合并
- `08_download_qwen3_8b.py` — 学生模型下载

### Phase 2: chosen / rejected 生成
- `09_taskC_opus_chosen.py` — Task C: Opus 网关 async chunked
- `10_taskA_m2_chosen.py` — Task A: M2 vllm 8 shard
- `11_taskB_qwen3_rejected.py` — Task B: Qwen3-8B 自答 rejected
- `12_taskD_m2_missing_chosen.py` — Task D: M2 兜底 work-stealing queue
- `15_taskE_rerun_bad.py` — Task E: 重跑撞顶 4307 条 (按 max_tokens SOP, 加 finish_reason 字段)

### 数据后处理
- `13_filter_chosen.py` — token 反推 finish_reason 通用过滤
- `13_taskD_filter.py` — Task D 专用过滤
- `14_smoke_max_tokens.py` — max_tokens SOP smoke (跑全量前必走)
- `16_prepare_dpo_dataset.py` — Phase 4 DPO 训练数据 prep (4 chosen + 1 rejected → ms-swift 格式)

### DLC launcher
- `taskA_m2_dlc.sh` / `taskB_qwen3_dlc.sh` / `taskD_m2_dlc.sh` / `taskE_rerun_dlc.sh` — Phase 2 DLC launcher
- **`taskF_dpo_2node_dlc.sh`** — **Phase 4 DPO launcher (2 节点 × 8 卡 = 16 卡)**

### audit 工具
- `audit/compact.py` — 把 audit_1000.jsonl 压缩成 compact 文本 (audit 数据本身已 .gitignore)

## 关键工程教训 (这次会话踩的坑, 都已落 memory)

| 坑 | 修法 |
|---|---|
| Task A shard 0: multi-shard /tmp cp race | launcher 串行 pre-cp, 启动 shard 前文件已稳定 |
| vllm 0.17.1 finalize ERROR / DLC 误标 fail | 升 0.19.1 + launcher 用 staging 行数判 success |
| Task D id % 8 + 缺口非均匀 → 19 GPU-h stragglers | static sharding → work-stealing queue (mp.Queue 共享) |
| Task C 网关 retry 风暴 9h 烧 ¥2200 | 403 立即 return + staging 增量监控早停 |
| Task D max_tokens=3072 撞顶 15% (15,300 条) | 升 max_tokens=6400 (smoke p99 × 1.15) + staging 必带 finish_reason |
| **DPO 关 grad_ckpt 直接 OOM (谷值 ≠ 峰值)** | **8B + DPO + ctx 4K+ 必开 gradient_checkpointing** |
| NCCL_DEBUG=INFO 顶掉 tqdm 进度条 | launcher 加 `NCCL_DEBUG=WARN` |
| swift dataset_num_proc=1 默认 → preprocess 5 min | `--dataset_num_proc 8` (8x 加速) |

## Phase 4 DPO 训练配置 (稳定版)

```bash
# DLC 提交命令 (Worker count=2, Worker GPU=8, vllm 镜像)
bash /mnt/data/huangjiawei/scripts/taskF_dpo_2node_dlc.sh
```

| 项 | 值 |
|---|---|
| 框架 | ms-swift 4.2.3 (swift_pkgs PYTHONPATH 走) |
| 启动 | 外层 torchrun + `swift.cli.rlhf` (双节点必这样) |
| 学生 | Qwen3-8B-Instruct |
| Tuner | LoRA r=16 / α=64 / target=all-linear |
| β | 0.1 (KL 强度) |
| lr | 5e-7 (DPO 比 SFT 低 10×) |
| epoch | 2 (1360 step, global_bs=64) |
| max_length | 6144 |
| **gradient_checkpointing** | **true (必开, 关掉 OOM)** |
| save_steps | 200 |
| 资源 | 2 节点 × 8 卡 A800-80GB |
| 实测 step time | 17.2 s/it |
| 实测 ETA | ~6.5h (1360 × 17.2s) |

## Phase 4 训练信号 (3 次启动累计验证)

```
step 1   : loss=0.69  accuracies=0.0   margins=0.0    ← 完全随机
step 25  : loss=0.69  accuracies=0.38  margins=0.12   ← 开始学
step 60  : loss=0.43  accuracies=0.79  margins=0.87   ← 飞速学
预期 step 200+ : accuracies 90%+, margins 1.5+
```

DPO 学得快是因为:
1. M2/Opus 老师 vs Qwen3-8B 学生差距悬殊 (audit 480 对验证 chosen 真胜率 95%+)
2. β=0.1 KL 约束适中
3. LoRA r=16 容量够吃信号

## 后续

- Phase 4 跑完 (1360 step) → ckpt 输出
- Phase 5 评估: v2 seed_eval 题集 + Baichuan-M2 7 题对照
- 部署: RK3588 32GB / CM5 类手机大小专用医疗问答终端 (见 `project_medical_dpo_edge_terminal_2026_06_17.md`)
