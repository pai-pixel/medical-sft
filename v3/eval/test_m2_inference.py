import os, time, resource
# transformers 4.57.6 import 期递归遍历 models/ 泄漏 fd,默认 1024 必崩
# memory 实测: 65536 时好时坏 / 131072 仍 flaky / 1048576 反而崩(极端值雷)
# 取中间稳态 262144 (grpo_mcq.sh 用过,稳)
_TARGET = 262144
_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(_TARGET, _h), _h))
print("[m2-test] RLIMIT_NOFILE %s -> %s (hard %s)" % (_s, resource.getrlimit(resource.RLIMIT_NOFILE)[0], _h), flush=True)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/mnt/data/huangjiawei/models/Baichuan-M2-32B"
THINK_END = 151668  # </think>

QUESTIONS = [
    "六味地黄丸的组成药物有哪些？请逐一列出。",
    "桂枝加葛根汤的组成是什么？它与桂枝加芍药汤在组成和主治上有何区别？",
    "理中丸由哪些药物组成？功效与主治是什么？",
    "二陈汤的组成、功效和主治分别是什么？",
    "患者男，45岁，眩晕头胀、面红目赤、急躁易怒、口苦、舌红苔黄、脉弦数。请中医辨证，并给出治法与代表方剂。",
    "一名既往体健的成年人确诊为轻症社区获得性肺炎（门诊治疗），经验性抗生素首选是什么？请给出循证依据。",
    "我最近总头晕，应该吃什么药？",
]

print("[load] %s  loading model..." % time.strftime("%H:%M:%S"), flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
print("[load] done in %.0fs" % (time.time() - t0), flush=True)

for i, q in enumerate(QUESTIONS, 1):
    text = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True, thinking_mode="on")
    inp = tok([text], return_tensors="pt").to(model.device)
    t = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=2048)
    gen = out[0][len(inp.input_ids[0]):].tolist()
    try:
        idx = len(gen) - gen[::-1].index(THINK_END)
    except ValueError:
        idx = 0
    think = tok.decode(gen[:idx], skip_special_tokens=True).strip()
    ans = tok.decode(gen[idx:], skip_special_tokens=True).strip()
    print("\n" + "=" * 70, flush=True)
    print("Q%d (%.0fs, %d tok): %s" % (i, time.time() - t, len(gen), q), flush=True)
    print("--- 思考(截断1200) ---", flush=True)
    print(think[:1200], flush=True)
    print("--- 回答 ---", flush=True)
    print(ans, flush=True)
print("\n[done] %s" % time.strftime("%H:%M:%S"), flush=True)
