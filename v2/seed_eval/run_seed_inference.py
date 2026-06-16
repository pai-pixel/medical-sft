#!/usr/bin/env python3
"""v2 SFT 推理验证 - transformers 单卡 batch 模式

为啥不用 vLLM:
- vllm_env 装在 /mnt/data/huangjiawei/vllm_env (virtiofs)
- import vllm 会 stat 几百个 .so/.py, virtiofs stat 慢路径,实测 >10 min 没起来
- 改用 system python (/usr/local/lib) + transformers,2 秒就 import 完

输入: /tmp/seed_questions.jsonl
输出: /tmp/seed_inference_out.jsonl

环境: /usr/local/bin/python3 (transformers 4.57.6 + torch + cuda)
显卡: 单张 A800-80GB (8B bf16 ~16GB, KV cache 占用因 batch 而定)
推理: batch=8, 35 道分 5 batch 跑完
"""
import json
import os
import sys
import time

CKPT = "/tmp/ckpt-13144"
QUESTIONS = "/tmp/seed_questions.jsonl"
OUT = "/tmp/seed_inference_out.jsonl"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 1024

SYSTEM_TCM = """你是一位资深中医师，精通中医辨证论治、方剂、本草和经典医籍。请根据用户提问基于中医理论给出辨证分析和方剂建议。
重要声明：中医属于传统医学经验体系，大部分推荐基于历代医家经验和经典医籍，缺乏现代循证医学(RCT)级别的临床证据，仅作为传统医学参考，不能替代循证医学诊疗。建议患者结合现代医学诊断综合判断，重大疾病请就医。"""

SYSTEM_EBM = """你是一位严谨的循证医学(EBM)医师，基于现代医学证据(随机对照试验、Meta 分析、临床指南、专家共识)进行诊断和治疗推荐。
回答时遵循以下原则：
1. 优先引用循证证据(临床指南、专家共识、RCT 结论)
2. 对缺乏充分证据的情况，明确告知"目前缺乏循证医学证据，需进一步专业评估"
3. 不推荐缺乏循证证据的传统医学方剂
4. 严重病情建议患者就医诊治"""


def expand_questions(items):
    expanded = []
    for q in items:
        track = q["system_track"]
        if track == "DUAL":
            for t in ("EBM", "TCM"):
                expanded.append({**q, "_run_track": t})
        else:
            expanded.append({**q, "_run_track": track})
    return expanded


def main():
    log = lambda m: print(f"[t={time.strftime('%H:%M:%S')}] {m}", flush=True)

    log("loading questions...")
    with open(QUESTIONS) as f:
        items = [json.loads(l) for l in f if l.strip()]
    items = expand_questions(items)
    log(f"  -> {len(items)} inference items "
        f"(TCM={sum(1 for x in items if x['_run_track']=='TCM')}, "
        f"EBM={sum(1 for x in items if x['_run_track']=='EBM')})")

    log("importing torch + transformers ...")
    import torch
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    log(f"  torch {torch.__version__}, cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")

    log("loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # decoder-only batch 推理必须 left-pad,否则模型 attention 错位生成崩坏
    tok.padding_side = "left"

    log(f"loading model from {CKPT} (~2-3 min for 8B from cpfs)...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"  model loaded in {time.time()-t0:.1f}s; "
        f"GPU mem allocated = {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

    # 构造 chat-template prompts
    log("building prompts ...")
    prompts = []
    for q in items:
        sys_p = SYSTEM_TCM if q["_run_track"] == "TCM" else SYSTEM_EBM
        msgs = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": q["question"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    max_in = max(len(tok(p)["input_ids"]) for p in prompts)
    log(f"  max input tokens = {max_in}")

    # batch 推理
    log(f"generating ({BATCH_SIZE} per batch) ...")
    t0 = time.time()
    answers = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to("cuda:0")
        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.05,
            )
        # 切掉输入部分,只留生成
        gen = out_ids[:, enc["input_ids"].shape[1]:]
        for g in gen:
            text = tok.decode(g, skip_special_tokens=True).strip()
            answers.append(text)
        elapsed = time.time() - t0
        done = len(answers)
        log(f"  batch {i//BATCH_SIZE+1}: cumulative {done}/{len(prompts)} "
            f"in {elapsed:.1f}s ({done/elapsed:.2f} req/s)")

    log(f"writing {OUT} ...")
    with open(OUT, "w") as f:
        for q, ans in zip(items, answers):
            rec = {
                "id": q["id"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "system_track": q["system_track"],
                "run_track": q["_run_track"],
                "question": q["question"],
                "model_answer": ans,
                "reference_answer": q.get("reference_answer", ""),
                "key_points": q.get("key_points", []),
                "notes": q.get("notes", ""),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log("DONE.")


if __name__ == "__main__":
    main()
