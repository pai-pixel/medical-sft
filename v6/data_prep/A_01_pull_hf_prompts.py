"""A_01: 从 HF 公开 dataset 拉 seed prompts,normalize 到统一 schema。

三源:
- FreedomIntelligence/CMB train → 1k MCQ (带 A/B/C/D + GT)
- FreedomIntelligence/medical-o1-reasoning-SFT → 过滤 acute keyword 3k
- Suprit/ChatDoctor-HealthCareMagic-100k-CN (or fallback shibing624/medical) → 短问答 4k

输出统一 schema:
  {"id": "hf_<src>_<idx>", "prompt": str, "source": str, "category": str,
   "gt_answer": str|null, "meta": {原字段}}

坑防守 (memory feedback_hf_env_before_import.md):
  必须在 import huggingface_hub / datasets 之前设 env。
"""
import os
# 关键:必须在 import huggingface_hub / datasets 之前设置(feedback_hf_env_before_import.md)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ.pop("HF_ENDPOINT", None)  # 不走 hf-mirror,直连 hf.co

import json, re, hashlib, random
from pathlib import Path
from collections import Counter
from datasets import load_dataset

OUT_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v6/seed_prompts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 中文占比
CN_RE = re.compile(r"[一-鿿]")
def cn_ratio(s: str) -> float:
    if not s: return 0.0
    total = sum(1 for c in s if c.isalnum() or CN_RE.match(c))
    if total == 0: return 0.0
    return sum(1 for c in s if CN_RE.match(c)) / max(total, 1)


def pull_cmb(target=1000):
    """FreedomIntelligence/CMB train — MCQ 带 GT"""
    print(f"[cmb] loading FreedomIntelligence/CMB (CMB-Exam) ...", flush=True)
    try:
        ds = load_dataset("FreedomIntelligence/CMB", "CMB-Exam", split="train")
    except Exception as e:
        print(f"[cmb] CMB-Exam failed: {e}, try no-config", flush=True)
        try:
            ds = load_dataset("FreedomIntelligence/CMB", split="train")
        except Exception as e2:
            print(f"[cmb] no-config also failed: {e2}, give up", flush=True)
            return 0
    print(f"[cmb] total {len(ds)} rows, sampling {target}", flush=True)
    random.seed(42)
    idxs = random.sample(range(len(ds)), min(target * 3, len(ds)))
    rows = []
    for i in idxs:
        r = ds[i]
        q = r.get("question", "").strip()
        opts = r.get("option") or {}
        # option 可能是 dict {A:.., B:..} 或 list
        if isinstance(opts, dict):
            opt_lines = [f"{k}. {v}" for k, v in sorted(opts.items()) if v]
        elif isinstance(opts, list):
            opt_lines = [f"{chr(65+j)}. {v}" for j, v in enumerate(opts) if v]
        else:
            opt_lines = []
        if len(opt_lines) < 2:
            continue
        gt = r.get("answer", "").strip()
        if not gt or not q:
            continue
        prompt = q + "\n" + "\n".join(opt_lines)
        if cn_ratio(prompt) < 0.5:  # 保证 CN 占比
            continue
        rows.append({
            "id": f"hf_cmb_{i:06d}",
            "prompt": prompt,
            "source": "FreedomIntelligence/CMB",
            "category": "mcq",
            "gt_answer": gt,
            "meta": {
                "question_type": r.get("question_type", ""),
                "subject": r.get("subject", ""),
                "explanation": r.get("explanation", ""),
            },
        })
        if len(rows) >= target:
            break
    out = OUT_DIR / f"hf_cmb_train_{len(rows)}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[cmb] wrote {len(rows)} → {out}", flush=True)
    return len(rows)


ACUTE_KEYWORDS = re.compile(
    r"儿科|婴|幼儿|新生|孕|妊娠|产|急救|抢救|急诊|中毒|误服|误吞|急性|昏迷|休克|"
    r"惊厥|抽搐|窒息|溺水|骨折|大出血|烫伤|烧伤|触电|中暑|心梗|脑卒|癫痫"
)

def pull_medical_o1_acute(target=3000):
    """FreedomIntelligence/medical-o1-reasoning-SFT — 过滤 acute keyword"""
    print(f"[o1] loading FreedomIntelligence/medical-o1-reasoning-SFT (zh) ...", flush=True)
    # 该 dataset 有 zh/en 两个 config
    try:
        ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "zh", split="train")
    except Exception as e:
        print(f"[o1] zh config failed: {e}, try default", flush=True)
        ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", split="train")
    print(f"[o1] total {len(ds)}, filtering acute", flush=True)
    rows = []
    for i in range(len(ds)):
        r = ds[i]
        q = r.get("Question") or r.get("question") or ""
        q = q.strip()
        if not q or cn_ratio(q) < 0.5:
            continue
        if not ACUTE_KEYWORDS.search(q):
            continue
        rows.append({
            "id": f"hf_o1_{i:06d}",
            "prompt": q,
            "source": "FreedomIntelligence/medical-o1-reasoning-SFT",
            "category": "acute",
            "gt_answer": None,
            "meta": {},
        })
        if len(rows) >= target:
            break
    if len(rows) < target:
        # 补:不带 acute keyword 但 zh 的,拿来做 short(下面 pull_short 里也拉,这里主要图 acute)
        print(f"[o1] only found {len(rows)} acute, keeping what we have", flush=True)
    out = OUT_DIR / f"hf_medical_o1_acute_{len(rows)}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[o1] wrote {len(rows)} → {out}", flush=True)
    return len(rows)


def pull_short_qa(target=4000):
    """尝试多个短问答 dataset,fallback 到底"""
    candidates = [
        ("Suprit/ChatDoctor-HealthCareMagic-100k-CN", None, "input", "output"),
        ("shibing624/medical", "pretrain", "text", None),
        ("michaelwzhu/ChatMed_Consult_Dataset", None, "query", "response"),
    ]
    for name, config, q_field, a_field in candidates:
        try:
            print(f"[short] trying {name} ({config}) ...", flush=True)
            if config:
                ds = load_dataset(name, config, split="train")
            else:
                ds = load_dataset(name, split="train")
            print(f"[short] {name}: total {len(ds)}, fields={list(ds.features)}", flush=True)
            random.seed(42)
            idxs = random.sample(range(len(ds)), min(target * 3, len(ds)))
            rows = []
            for i in idxs:
                r = ds[i]
                q = (r.get(q_field) or "").strip()
                if not q or cn_ratio(q) < 0.5:
                    continue
                if len(q) < 8 or len(q) > 150:
                    continue
                rows.append({
                    "id": f"hf_short_{i:06d}",
                    "prompt": q,
                    "source": name,
                    "category": "short",
                    "gt_answer": None,
                    "meta": {"ref_answer": (r.get(a_field) or "").strip()[:500]} if a_field else {},
                })
                if len(rows) >= target:
                    break
            if len(rows) >= target // 2:
                out = OUT_DIR / f"hf_short_qa_{len(rows)}.jsonl"
                with out.open("w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"[short] wrote {len(rows)} → {out}", flush=True)
                return len(rows)
            else:
                print(f"[short] only {len(rows)}, try next", flush=True)
        except Exception as e:
            print(f"[short] {name} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
    print(f"[short] all candidates failed, wrote 0", flush=True)
    return 0


if __name__ == "__main__":
    stats = {}
    stats["cmb"] = pull_cmb(1000)
    stats["o1_acute"] = pull_medical_o1_acute(3000)
    stats["short"] = pull_short_qa(4000)
    print(f"\n=== A_01 summary ===\n{json.dumps(stats, indent=2, ensure_ascii=False)}", flush=True)
