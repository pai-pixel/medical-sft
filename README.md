# Medical SFT (中西医医疗对话模型微调工程)

> 基于 Qwen3-VL-8B-Instruct 的中西医分轨医疗对话模型 SFT 工程。  
> **当前状态**:v2 训练中(2026-06-08 启动,预计 2026-06-10 完成)。

## 项目定位

中西医分轨医疗对话**对话风格 backbone**,**非终态产品**。

严肃医疗级产品需要四件事:
1. ✅ **SFT(对话风格层)** — 本工程在做
2. ❌ **RAG(证据检索层)** — 待建
3. ❌ **拒答(硬编码门控)** — 待建
4. ❌ **审计(全链路日志)** — 待建

仅有 SFT 训出来的模型本质是模仿语料风格,**不会真正"循证"**,不能直接部署给医生/患者使用。

## v1 vs v2

| | v1 (废) | **v2 (当前)** |
|---|---|---|
| 基模 | Qwen3-VL-2B | **Qwen3-VL-8B-Instruct** |
| 训练方式 | LoRA r=8(误用) | **full SFT** (`--tuner_type full`) |
| 数据 | 11.3 万(单源 ShenNong) | **168 万(13 源,EBM 64% + TCM 36%)** |
| max_length | 2048 | **4096** |
| epochs | 3 | **2** |
| zero | 2 | **3** |
| lr | 1e-5 | **5e-6** |
| **system prompt** | 单一中医师 | **双轨自动切换**(EBM/TCM) |

### v1 教训
- 没显式 `--tuner_type full`,swift 4.2.3 默认走 LoRA r=8,容量太小
- 数据来源 ShenNong 自身有错(理中丸成分错、二陈"汉药"等),被模型学到错的方剂
- 8 道中医测题:全对 0,有重大错误 5(六味地黄丸成分错、桂枝加葛根写成加芍药)

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
├── README.md                    本文件
├── docs/                        设计文档
├── v1/                          v1 工程(已废,作参考)
│   ├── download_shennong_tcm.sh
│   ├── normalize_to_swift.py    单源 normalize
│   ├── sft_qwen3vl_2b_tcm.sh    2B + LoRA(误)启动
│   └── test_inference.sh        推理脚本
└── v2/                          v2 工程(当前)
    ├── sft_v2_dlc.sh            8B + full SFT + zero3 DLC 启动
    ├── normalize_v2.py          多源 normalize + 配额抽样 + 双轨 system + HuatuoGPT 格式
    └── v2_config.json           13 源数据配置(每源标 track + system)
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

## 下一步路线(等 v2 训完)

1. **W1 推理验证** — 中医 + EBM 各 8 道题,看双轨切换是否生效,跟 v1 对比错误率
2. **W2-3 选定专科** — 生殖医学 / 肿瘤 / 罕见病 / 慢病 选一个做 RAG MVP
3. **W4-5 RAG 证据库** — MedRAG 范式 + BM25 + BGE-M3 + reranker,语料用专科指南/共识/PubMed/DailyMed
4. **W6-7 EMR 结构化能力** — MTS-Dialog / ACI-BENCH / IMCS-21 范式补
5. **W8-10 端到端 PoC + 拒答数据自建**
6. **W11-12 找客户(医生/医院信息科/药企医学部)看 demo**

## 参考资料

- [ms-swift](https://github.com/modelscope/ms-swift) 训练框架
- [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) RAG 参考架构
- [CBLUE](https://github.com/CBLUEbenchmark/CBLUE) 中文医学 NLU benchmark
- [CMB](https://github.com/FreedomIntelligence/CMB) 中文医学 benchmark
- [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) 医患对话 → 病历

## License

工程代码 MIT。**数据集各自的 license 见 HF 页面,本仓库不包含数据本身。**
