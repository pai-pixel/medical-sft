---
language:
- zh
license: other
library_name: transformers
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-8B
tags:
- medical
- chinese-medicine
- tcm
- sft
- qwen3
---

# medical-sft-v4-qwen3-8b

基于 Qwen3-8B 的中西医医疗对话模型 v4(2026-06-20 完工)。

## 训练摘要

| 项 | 值 |
|---|---|
| 基模 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (chat 版) |
| 方法 | full SFT(swift 4.2.3) |
| 数据 | 37,353 条(配套数据集 `medical-sft-v4-data`) |
| 拓扑 | DLC 4 节点 × 8 GPU = 32 卡 |
| 超参 | lr 5e-6 / 2 epoch / max_len 4096 / global_bs 128 / zero3 |
| 训练时间 | 1h 14m |
| 收敛 | loss 1.39 → 0.75,token_acc 0.654 → 0.764 |

## 评估结果(1390 题,跟 base / v2 / v3-DPO / M2-32B 对照)

| 指标 | qwen3-8b base | v2 SFT | **v4 SFT** | M2-32B |
|---|---|---|---|---|
| CMB 235 | 72.3% | 71.0% | **69.8%**(-2.5%) | 47.2% |
| MedQA-CN 1000 | 80.8% | 80.1% | **80.1%**(-0.7%) | 47.1% |
| C-Eval 通识 50 | 80% | 92% | **84%**(+4%) | 58% |
| open_v2_seed judge | 4.73 | 4.60 | **5.0** | 5.0 |
| **open_new_30 临床决策** | 4.73 | **4.20** | **5.0** | 5.0 |

**核心成果**: 修住 v2 "回避临床判断" 硬伤(open_new_30 4.20 → 5.0,跟 M2-32B 同档),教材方剂剂量正确(六味地黄丸/桂枝汤/麻黄汤/理中丸 100% 对)。

## 用法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "shdkahjkda/medical-sft-v4-qwen3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)

system = "你是一位资深临床医师,精通中医辨证施治与现代循证医学..."
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": "六味地黄丸的标准组成、剂量、功用、主治是?"},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    enable_thinking=False,  # v4 切除 thinking,直接出 final answer
    return_tensors="pt",
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=2048, temperature=0.3, top_p=0.9)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

## 已知局限

- **切除了 thinking 段**: v4 学到 M2 final answer 风格,但没继承 reasoning 链路,**训练分布外场景可能死记硬背**
- **客观题略低于 base**: -2.5% CMB 在统计置信区间内,long-form SFT 副作用
- **复杂临床推理弱于 M2**: M2 thinking 模式下能多步辨证,v4 在罕见病/疑难杂症仍是短板
- **D2/D3/D4 西药/急救剂量数据待二核**(NMPA 原厂说明书),当前仅有 D1 教材方剂

## 训练数据

详见配套数据集 [shdkahjkda/medical-sft-v4-data](https://huggingface.co/datasets/shdkahjkda/medical-sft-v4-data)。

## License

- 基模 Qwen3-8B: Apache 2.0
- 训练数据: 详见配套数据集 README
- 二次合成(Opus 蒸馏部分): **Anthropic ToS 商用风险**

## 配套代码

GitHub: https://github.com/pai-pixel/medical-sft (`v4/` 子目录)

## Citation

```bibtex
@misc{medical-sft-v4-2026,
  title={medical-sft-v4-qwen3-8b: 中西医医疗对话模型},
  year={2026},
  url={https://huggingface.co/shdkahjkda/medical-sft-v4-qwen3-8b}
}
```
