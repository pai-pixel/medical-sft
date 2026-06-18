# Medical SFT (中西医医疗对话模型微调工程)

> 基于 Qwen3-VL/Qwen3 的中西医分轨医疗对话模型,**四阶段迭代**:v1 LoRA SFT 初探 → v2 大规模双轨 full SFT → v3 GRPO 强化学习 → **v3 DPO 双老师蒸馏**。

**当前状态**(2026-06-18):
- v1 (2B+LoRA SFT) — 已废,8 道方剂题全错
- v2 (8B full SFT 168 万双轨) — 已冷冻,实测推理差(2026-06-11)
- v3 GRPO (LoRA + GRPO + RLVR) — 工程闭环跑通,RL 效果零(详见 `docs/v3-grpo.md`)
- **v3 DPO (Qwen3-8B + LoRA + 双老师蒸馏)** — **工程闭环跑通**,数据 43k 配对 + 训练已 3 次启动验证(详见 `v3/dpo/README.md`)

横向对照:跟开源医疗 SOTA **Baichuan-M2-32B** 同题对比 7/7 全过,印证了 v2 "蒸馏数据 SFT 学不到真推理" 的判断(详见 `docs/baichuan-m2-comparison.md`),并据此把 M2 引入为 v3 DPO 的 TCM 老师.

## 项目定位

中西医分轨医疗对话**对话风格 backbone**,**非终态产品**。

严肃医疗级产品需要四件事:
1. ✅ **SFT(对话风格层)** — 本工程在做
2. ❌ **RAG(证据检索层)** — 待建
3. ❌ **拒答(硬编码门控)** — 待建
4. ❌ **审计(全链路日志)** — 待建

仅有 SFT 训出来的模型本质是模仿语料风格,**不会真正"循证"**,不能直接部署给医生/患者使用。

## 配套 HuggingFace Datasets

代码在 GitHub,**完整数据集发布在 HuggingFace**(全 private,需要 token)。

| GitHub 子目录 | HF Dataset (private) | 规模 |
|---|---|---|
| `v2/` (full SFT 168万) | [`shdkahjkda/medical-sft-v2`](https://huggingface.co/datasets/shdkahjkda/medical-sft-v2) | 2.6 GB / 1.68M 条 |
| `v3/` (GRPO 1067 条 CMMLU) | [`shdkahjkda/medical-mcq-v3-grpo`](https://huggingface.co/datasets/shdkahjkda/medical-mcq-v3-grpo) | 600 KB / 1067 条 |
| **`v3/dpo/`** (DPO 双老师蒸馏 43,488 配对) | [`shdkahjkda/medical-dpo-43k`](https://huggingface.co/datasets/shdkahjkda/medical-dpo-43k) | 1.17 GB / 43,488 对 |

数据闭环:本仓库**只含代码 + 1000 对 audit 抽样**(`v3/dpo/audit/audit_1000.jsonl`),完整数据需从 HF 拉取:

```python
from datasets import load_dataset
ds = load_dataset("shdkahjkda/medical-dpo-43k", split="train")  # 43,488 DPO 配对
```

## v1 / v2 / v3 三阶段对比

| | v1 (废) | v2 (冷冻) | **v3 (GRPO 跑通)** |
|---|---|---|---|
| 基模 | Qwen3-VL-2B | Qwen3-VL-8B-Instruct | v2 ckpt-13144 |
| 训练方式 | LoRA r=8(误用) | full SFT (`--tuner_type full`) | **LoRA + GRPO + RLVR** |
| 数据 | 11.3 万(单源 ShenNong) | 168 万(13 源,EBM 64% + TCM 36%) | CMMLU 医学 8 subject 1067 条 |
| max_length | 2048 | 4096 | 2048(prompt)+ 1024(completion) |
| 算力 | hjw 单机 | 2 节点 × 8 GPU = 16 卡 | 1 节点 × 8 卡 |
| 训练耗时 | 短 | ~39 小时 | 1h 11m / 532 step |
| 学习率 | 1e-5 | 5e-6 | 1e-6 |
| **system prompt** | 单一中医师 | 双轨自动切换(EBM/TCM) | 跟随 v2 |
| **结果** | 8/8 错 | 推理弱 | 工程闭环 ✅ 效果零 ❌ |

### v1 教训
- 没显式 `--tuner_type full`,swift 4.2.3 默认走 LoRA r=8,容量太小
- 数据来源 ShenNong 自身有错(理中丸成分错、二陈"汉药"等),被模型学到错的方剂
- 8 道中医测题:全对 0,有重大错误 5(六味地黄丸成分错、桂枝加葛根写成加芍药)

### v2 教训
- ChatGPT 蒸馏的"EBM"数据本身不是真证据,SFT 只能学到风格不是事实
- 8B + 168 万双轨 + full SFT 仍然推不出真循证推理 → SFT 单阶段方法本身天花板
- 据此 2026-06-11 冷冻 v2,转向 RL 路径

### v3 教训
- GRPO 不看 loss(在 0 附近震荡是正常),看 reward / mcq_acc 曲线
- 起跑就 acc 0.72 → 接近 reward 天花板 → RL 啃不动
- kl ≈ 0.001 全程 → 更新强度太小 → 模型本质没在学
- 25-50% 的 prompt 组,8 generation 全对或全错 → 组内 advantage = 0 → 训练零贡献

## v2 关键创新:双轨 system prompt

每条数据按来源挂对应 system,模型学到 **system → 风格** 强映射。推理时前端切 system 走对应轨。

**EBM 轨**(西医循证):shibing624/medical, ChatMed_Consult, CMtMedQA, HuatuoGPT, Huatuo26M-Lite
```
你是一位严谨的循证医学(EBM)医师,基于现代医学证据(RCT、Meta、临床指南)进行诊断和治疗推荐。
缺乏充分证据时明确告知"目前缺乏循证医学证据,需进一步专业评估"。
不推荐缺乏循证证据的传统医学方剂。
```

**TCM 轨**(中医辨证):ShenNong + SylvanL 全套(Syndrome/Prescription/DiseaseDiagnosed/MedKnowledge/StructGeneral)
```
你是一位资深中医师,精通中医辨证论治、方剂、本草和经典医籍。
重要声明:中医属于传统医学经验体系,大部分推荐基于历代医家经验和经典医籍,
缺乏现代循证医学(RCT)级别的临床证据,仅作为传统医学参考,不能替代循证医学诊疗。
```

## 数据集来源

| 数据集 | 类型 | 全量 | v2 用量 | License |
|---|---|---|---|---|
| `michaelwzhu/ShenNong_TCM_Dataset` | 中医 QA | 113K | 107K(过滤后) | 见 HF |
| `SylvanL/Traditional-Chinese-Medicine-Dataset-SFT` | 中医医案+处方+知识 | ~3.7M | 502K(配额抽样) | apache-2.0 |
| `shibing624/medical` | 中文医疗 | 1.95M | 200K | 见 HF |
| `michaelwzhu/ChatMed_Consult_Dataset` | 中文在线问诊 | 549K | 527K(过滤后) | 见 HF |
| `Suprit/CMtMedQA` | 中文医学 QA | 68K | 全量 | 见 HF |
| `FreedomIntelligence/HuatuoGPT-sft-data-v1` | 医患多轮对话 | 226K | 100K(配额) | 见 HF |
| `FreedomIntelligence/Huatuo26M-Lite` | 中文医疗 QA(精选) | 178K | 全量 | 见 HF |

**许可声明**:本仓库**不**包含数据本身,仅包含数据下载/处理脚本。商业使用前请自行确认每个数据集的下游许可。

## 训练配置(v2)

```yaml
模型: Qwen3-VL-8B-Instruct
训练方式: full SFT
分布式: 2 节点 × 8 GPU = 16 卡 (A100 80G)
zero stage: 3
max_length: 4096
per_device_batch_size: 2
gradient_accumulation_steps: 8
global_batch_size: 256  # 2 × 8 × 16
num_train_epochs: 2
learning_rate: 5e-6
lr_scheduler: cosine
warmup_ratio: 0.03
gradient_checkpointing: true
mixed_precision: bf16

总 step: 13,144  (1,682,199 × 2 / 256)
预计 wall time: ~39 小时
```

## 文件结构

```
medical_sft/
├── README.md                       本文件
├── docs/                           设计文档 + 阶段沉淀
│   ├── lessons-learned.md          5 个迭代踩坑(v1→v2)
│   ├── v3-grpo.md                  v3 GRPO 跑通记录 + 失败原因 + 简历口径
│   ├── baichuan-m2-comparison.md   跟开源医疗 SOTA 7 题对照评估
│   └── resume-pipeline.md          简历级三阶段流水线大纲
├── v1/                             v1 工程(已废,作参考)
│   ├── download_shennong_tcm.sh
│   ├── normalize_to_swift.py       单源 normalize
│   ├── sft_qwen3vl_2b_tcm.sh       2B + LoRA(误)启动
│   └── test_inference.sh           推理脚本
├── v2/                             v2 工程(已冷冻)
│   ├── sft_v2_dlc.sh               8B + full SFT + zero3 DLC 启动
│   ├── normalize_v2.py             多源 normalize + 配额抽样 + 双轨 system
│   ├── v2_config.json              13 源数据配置
│   └── seed_eval/                  v2 SFT 完种子题集评测
│       ├── system_prompts.py
│       ├── run_seed_inference.py
│       ├── seed_questions.jsonl    种子题集
│       ├── seed_report.html        评测报告
│       └── ...
└── v3/                             v3 工程(GRPO + DPO)
    ├── README.md                   v3 一句话定调 + 算法配方 + 文件清单
    ├── mcq_reward.py               GRPO 自定义 reward 插件
    ├── prep_medical_mcq.py         CMMLU 医学子集 → train/heldout
    ├── find_medical_mcq.py         CMMLU schema 探查
    ├── run_grpo.py                 GRPO fd-wrapper
    ├── grpo_mcq.sh / grpo_mcq_dlc.sh   GRPO launcher
    ├── eval/                       开源医疗对照评估
    │   ├── test_m2_inference.py    Baichuan-M2-32B 7 题对照
    │   ├── download_baichuan_m2.py
    │   └── bcm2_probe.py
    └── dpo/                        ★ DPO 双老师蒸馏 (本会话产出)
        ├── README.md               双老师 + 4 件套数据 + Phase 4 launcher 详解
        ├── 01-08_*.py              prompt 池构建 (v2 抽样 + Sonnet 过滤 + Opus 自生成)
        ├── 09_taskC_opus_chosen.py     Task C: Opus 网关 EBM/General chosen
        ├── 10_taskA_m2_chosen.py       Task A: M2 vllm TCM chosen
        ├── 11_taskB_qwen3_rejected.py  Task B: Qwen3-8B 自答 rejected
        ├── 12_taskD_m2_missing_chosen.py  Task D: M2 兜底 work-stealing queue
        ├── 13_filter_chosen.py     token 反推 finish_reason 通用过滤
        ├── 14_smoke_max_tokens.py  max_tokens SOP smoke 工具
        ├── 15_taskE_rerun_bad.py   Task E: 重跑撞顶 4307 (含 finish_reason 字段)
        ├── 16_prepare_dpo_dataset.py   4 chosen + rejected → ms-swift DPO 格式
        ├── taskA-E_*_dlc.sh        Phase 2 DLC launcher
        ├── taskF_dpo_2node_dlc.sh  ★ Phase 4 DPO 训练 launcher (2 节点 × 8 卡)
        └── audit/compact.py        1000 对 chosen vs rejected audit 工具
```

## 部署步骤(v2)

### 0. 环境准备(在 ms-swift 兼容环境下)

需要:
- `ms-swift==4.2.3`(我们用的是 huangjiawei 的 fork,`--tuner_type` 不是 `--train_type`)
- `transformers>=4.57`
- `torch>=2.0`
- `deepspeed`

### 1. 数据下载

参考 `v2/v2_config.json` 里 13 个数据集的 HF repo id,逐一下载到本地。建议用 hf-mirror.com + 直连 huggingface.co 跨境对比择速。

⚠️ HF 下载坑(已踩):
- hf-mirror.com 实际 302 重定向到美国 cas-bridge.xethub.hf.co,跨境慢且不稳
- aria2c 多并发被 xethub 限流
- 单线程 wget 直连 huggingface.co(PAI/DSW 节点)~3 MB/s 反而最稳

### 2. Normalize + 合并

```bash
python3 v2/normalize_v2.py \
    --config v2/v2_config.json \
    --output /path/to/train_v2.jsonl
shuf train_v2.jsonl -o train_v2_shuf.jsonl
```

输出格式(swift `messages` 兼容):
```json
{
  "system": "你是一位严谨的循证医学(EBM)医师...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "_source": "shibing624_medical_zh",
  "_track": "EBM"
}
```

### 3. 训练(阿里云 PAI DLC)

把 `v2/sft_v2_dlc.sh` 放到节点能看到的路径(我们这里是 `/mnt/data/huangjiawei/scripts/`),改其中 `MODEL` / `DATA` 路径,然后 DLC 启动命令:

```bash
bash /path/to/sft_v2_dlc.sh
```

DLC 自动注入 `RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT`,脚本会接住。

### 4. 推理

待 v2 训完补充。可用 `transformers` + `Qwen3VLForConditionalGeneration` + 切换 system 测试两轨切换效果。

## 已知盲点

| | 现状 | 解决方向 |
|---|---|---|
| **不会真正循证** | SFT 学的是风格不是事实 | 加 RAG 做证据检索 + 引用 |
| **EBM 数据不真循证** | Huatuo/ChatMed 等都是 ChatGPT 蒸馏,不是 RCT | 必须配真证据库(PubMed/DailyMed/指南) |
| **没拒答机制** | 模型遇到不确定也会编 | 硬编码 `evidence_gate()`,不在 prompt 层 |
| **数据本身有错** | ShenNong 113K 里有方剂瞎编 | v2 已过滤明显错条,深层错误靠 SylvanL 临床数据"压过" |
| **VLM 部分没训** | ViT 冻结,只训 language_model | 后期 stage-2 视觉 SFT 可上(舌诊/影像)|

## 下一步路线(已暂停,留作未来)

v3 GRPO 工程闭环跑完 + 跟 M2 横向对照之后,医疗方向**整体冷冻**(简历项目使命已达成)。若以后重启:

1. **真做出 RL 提升** — `docs/v3-grpo.md` 给了 ABC 三种修法(加猛火 / 换更难数据 / 换更弱起点)
2. **走 M2 路线** — Apache 2.0 license 允许,可以从 GPTQ-Int4 单 4090 部署(零成本),或黑盒/白盒蒸馏到 Qwen2.5-7B 移动端(详见 `docs/baichuan-m2-comparison.md`)
3. **补四件套** — SFT 只占严肃医疗产品的 1/4,RAG / 拒答 / 审计 三件全待建

## 参考资料

- [ms-swift](https://github.com/modelscope/ms-swift) 训练框架
- [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) RAG 参考架构
- [CBLUE](https://github.com/CBLUEbenchmark/CBLUE) 中文医学 NLU benchmark
- [CMB](https://github.com/FreedomIntelligence/CMB) 中文医学 benchmark
- [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) 医患对话 → 病历

## License

工程代码 MIT。**数据集各自的 license 见 HF 页面,本仓库不包含数据本身。**
