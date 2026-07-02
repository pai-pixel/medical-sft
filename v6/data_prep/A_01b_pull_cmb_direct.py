"""A_01b: 单独拉 CMB train (绕开 datasets library schema 校验)。

CMB 的 train/test 有 schema 不一致(train 有 answer 无 id, test 反之),datasets 强制校验挂。
直接 hf_hub_download 抓 CMB-train-merge.json,手 parse。
"""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ.pop("HF_ENDPOINT", None)

import json, re, random
from pathlib import Path
from huggingface_hub import hf_hub_download

OUT_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v6/seed_prompts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CN_RE = re.compile(r"[一-鿿]")
def cn_ratio(s: str) -> float:
    if not s: return 0.0
    total = sum(1 for c in s if c.isalnum() or CN_RE.match(c))
    if total == 0: return 0.0
    return sum(1 for c in s if CN_RE.match(c)) / max(total, 1)


def main():
    target = 1000
    print(f"[cmb] hf_hub_download CMB-Exam/CMB-train/CMB-train-merge.json ...", flush=True)
    fp = hf_hub_download(
        repo_id="FreedomIntelligence/CMB",
        filename="CMB-Exam/CMB-train/CMB-train-merge.json",
        repo_type="dataset",
    )
    print(f"[cmb] downloaded {fp}", flush=True)
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[cmb] total {len(data)} raw rows", flush=True)

    random.seed(42)
    random.shuffle(data)

    rows = []
    for i, r in enumerate(data):
        q = (r.get("question") or "").strip()
        opts = r.get("option") or {}
        if isinstance(opts, dict):
            opt_lines = [f"{k}. {v}" for k, v in sorted(opts.items()) if v]
        elif isinstance(opts, list):
            opt_lines = [f"{chr(65+j)}. {v}" for j, v in enumerate(opts) if v]
        else:
            opt_lines = []
        if len(opt_lines) < 2 or not q:
            continue
        gt = (r.get("answer") or "").strip()
        if not gt:
            continue
        # gt 可能是完整选项文本, 只留字母
        gt_letter = gt[0].upper() if gt[0].upper() in "ABCDE" else ""
        if not gt_letter:
            continue
        prompt = q + "\n" + "\n".join(opt_lines)
        if cn_ratio(prompt) < 0.5:
            continue
        rows.append({
            "id": f"hf_cmb_{i:06d}",
            "prompt": prompt,
            "source": "FreedomIntelligence/CMB",
            "category": "mcq",
            "gt_answer": gt_letter,
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


if __name__ == "__main__":
    main()
