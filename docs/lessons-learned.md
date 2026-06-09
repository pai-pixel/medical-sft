# Lessons Learned — 踩坑实录

按优先级整理,5 条全是这次 v1→v2 迭代中真实踩过的坑。

---

## 1. ms-swift 4.2.3 用 `--tuner_type` 不是 `--train_type`

`/mnt/data/huangjiawei/swift_pkgs/`(swift 4.2.3,可能是个 fork 或旧版本)的全量微调开关叫 **`--tuner_type full`**。

写错的后果:
- `--train_type full` → `ValueError: remaining_argv: ['--train_type', 'full']`
- 不写任何开关 → **默认走 LoRA r=8**,会**悄悄训成 LoRA**(我们 v1 就是这样训歪的,以为是 full)

支持的值:`lora / longlora / adalora / llamapro / adapter / vera / boft / fourierft / reft / bone / full`

源码位置:`swift_pkgs/swift/pipelines/train/tuner.py:164,372`

**怎么验证训出来的真是 full SFT**:
- ckpt 目录里有 `model-*.safetensors` 4 个分片(完整模型) → ✅ full
- ckpt 目录里只有 `adapter_config.json` + `adapter_model.safetensors` 一个小文件 → ❌ LoRA
- 训练日志开头 `[INFO:swift] model_parameter_info: ... Trainable [93.42%]` — 90%+ 可训练 = full

**跨版本备忘**:官方 ms-swift 新版本(GitHub modelscope/ms-swift)用 `--train_type`,我们手上这个 4.2.3 fork 还是旧名 `--tuner_type`。给别人脚本要看 swift 来源决定参数名。

---

## 2. swift `--packing true` 必须配 flash_attn

我们 vllm_env 没装 flash_attn,加上 `--packing true` 启动直接报:

```
ValueError: The "packing" feature requires a flash attention implementation.
Please use one of: "flash_attn", "flash_attention_2", "flash_attention_3", "flash_attention_4".
```

**Why**:swift 4.2.3 的 packing 实现把多样本拼到 max_length 后,需要 flash_attn 的 varlen 接口靠 attention mask 隔开样本(防止跨样本 attend)。SDPA 不支持这个语义,所以强制检查。

**怎么办**:
- 想用 packing → 先在环境里装 flash-attn(~30 分钟编译,看 CUDA 兼容)
- 不想装 → 删 `--packing true`,GPU 利用率从 95% 退回 80%,wall time 从 ~25h 涨到 ~40h
- v2 为了快直接砍了 packing,保留 `--dataloader_num_workers 8 --dataloader_pin_memory true`(独立加速,不依赖 fa)

**同源关联**:`--padding_free true` 也走同一检查。

源码位置:`swift_pkgs/swift/arguments/sft_args.py:185-194`

---

## 3. HuatuoGPT-v1 字段是 `{"data": ["问:...", "答:..."]}` 非标准格式

`FreedomIntelligence/HuatuoGPT-sft-data-v1` 不用 `instruction/input/output`,也不用 `query/response`,而是:

```json
{"data": ["问：xxx？\n", "答：xxx。"]}
```

通用 normalize 脚本(认 `query/instruction/...`)会**全 dropped 100%**(我们这次 99749 条全失败,kept=0)。

**修法**:

```python
if "data" in rec and isinstance(rec["data"], list) and len(rec["data"]) >= 2:
    user = re.sub(r"^问[:：]\s*", "", rec["data"][0]).strip()
    asst = re.sub(r"^答[:：]\s*", "", rec["data"][1]).strip()
    if user and asst:
        return {"system": ..., "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst}]}
```

**同源 Huatuo26M-Lite 不一样**:`format_data.jsonl` 是标准 `{"questions": "...", "answers": "..."}`,不要混。

**通用提醒**:加任何中文医疗数据集前,**先 head -1 看一眼字段长什么样**,别假设它跟其他源一致。

---

## 4. hf-mirror 是假镜像,阿里 PAI 节点直连 hf.co 反而最快

我们 v2 数据下载实测(2026-06-08):

| 路径 | 单线程速度 | 备注 |
|---|---|---|
| `hf-mirror.com` | **60-300 KB/s,且 8s timeout** | 实际不是真镜像,302 转 us-east 跨境 |
| `huggingface.co` 直连 | **3-6 MB/s 稳定** | 阿里 PAI 内部白名单,实测最快 |
| aria2c -x 16 多并发(任一 endpoint) | **0 KB/s,被限流** | xethub CDN 限单 IP 多连接 |

**Why**:hf-mirror.com 表面是 mirror,实际只代理 API,文件下载 302 重定向到 `cas-bridge.xethub.hf.co/xet-bridge-us/...`(美国 us-east-1),最终还是跨境。再叠加 hf-mirror 服务时挂(curl -I 8 秒超时)。

阿里云 PAI / DSW / DLC 节点对 huggingface.co 有专线/白名单加速(实测 hf.co 直连 302 < 1s,后续真实下载 3-6 MB/s),反而绕过 hf-mirror 更快更稳。

**正确姿势**:

```bash
# 6 个文件并行 wget(单线程 + 跨进程并行),聚合 ~6-15 MB/s
nohup wget -c -q --tries=20 --timeout=120 \
  "https://huggingface.co/datasets/$REPO/resolve/main/$FILE" \
  -O "$OUT" > /tmp/dl.log 2>&1 &
```

- ❌ 不要再设 `HF_ENDPOINT=https://hf-mirror.com`
- ❌ 不要 aria2c 多并发(单文件 -x >4 大概率被 xethub 限流后变 0)
- ✅ wget 直连 hf.co + 跨进程并行

**aria2c sparse prealloc 陷阱**:aria2c 启动会预分配 sparse 文件占满最终 size,`du -sh` 立刻显示 100% 但实际 0 字节真数据。看真实进度要看 `.aria2` 控制文件大小或网卡 RX 流量(`/proc/net/dev`)。被限流时进程在但下载停滞,容易误以为下完了。

---

## 5. 加载 Qwen3-VL 必须用 `Qwen3VLForConditionalGeneration`

```python
# ❌ 卡几分钟到永久(D 状态 wait_on_page_bit_common)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True)
```

```python
# ✅ 1-2 分钟正常加载
from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16)
```

**Why**:Qwen3-VL 的 `config.json` 里 `architectures: ["Qwen3VLForConditionalGeneration"]`,Auto 在 dispatch 时找不到 image-text-to-text 任务的 CausalLM 映射,可能进入 trust_remote_code 远程加载 / sentinel 阻塞 / fallback 循环路径。具体表现:进程在 Sl/Dl 状态,wchan 是 `wait_on_page_bit_common`(virtiofs 等页缓存)或 `request_wait_answer`(FUSE 等响应),`tokenizer = AutoTokenizer.from_pretrained(...)` 都打不出来。

**通用规则**:
- Qwen3-VL 系列(2B / 8B / 32B)→ `Qwen3VLForConditionalGeneration`
- Qwen2-VL 系列 → `Qwen2VLForConditionalGeneration`
- 通用兜底 → `AutoModelForVision2Seq` 或 `AutoModelForImageTextToText`(transformers 4.50+)
- **永远不要用 `AutoModelForCausalLM` 加载 VLM**

`AutoTokenizer.from_pretrained` 对 VLM 没问题,因为 tokenizer 跟具体模型类无关。

**完整推理 LoRA adapter 模板**:

```python
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(base)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base, dtype=torch.bfloat16, device_map="cuda:0"
)
model = PeftModel.from_pretrained(model, ckpt_path)
model.eval()
```

**DSW 上还要绕 virtiofs**:`/mnt/data/huangjiawei/vllm_env/bin/python` 加载 transformers 会因 virtiofs page cache cold 卡几分钟。改用 system python `/usr/local/bin/python`(容器 overlay fs)立即起来。vllm_env 仅在 DLC 训练 GPU 节点上稳定(那边路径热)。
