"""A_02: 从 v4 D1 296 条方剂教材直接抽 seed → 1k(不够就重复采样成不同 prompt 变体)。

v4 D1 是《方剂学》第十版抽的方剂条目, schema:
  {"id": "D1_xxx", "src": ..., "system": ..., "messages": [{"role":"user","content":<方剂问题>}, {"role":"assistant","content":<教材答案>}]}

策略:
- 直接抽 prompt 段(user content),不重训只做 seed
- 296 条 x ~3.4 变体 = 1000
- 变体:同一方剂,让 Opus 换角度问("XX方的组成/功用/主治/加减化裁分别是啥?")
- 输出: seed_prompts/d1_textbook_tcm_seed_1k.jsonl
"""
import json, random
from pathlib import Path
from collections import Counter

D1_IN = Path("/mnt/data/huangjiawei/datasets_local/medical_v4/train/v4_D1_chosen.jsonl")
OUT = Path("/mnt/data/huangjiawei/datasets_local/medical_v6/seed_prompts/d1_textbook_tcm_seed.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "{prompt}",  # 原样
    "请详细讲解{fangji}的出处、组成、功用、主治与方义。",
    "{fangji}的加减化裁有哪些常见变化?请举例说明。",
    "{fangji}适用于哪些证型?配伍原则是什么?",
    "{fangji}的临床应用要点和使用注意有哪些?",
]

FANGJI_RE = __import__("re").compile(r"[一-鿿]{2,8}(?:汤|散|丸|饮|膏|丹|片|煎|方)")


def extract_fangji_name(prompt: str) -> str:
    m = FANGJI_RE.search(prompt)
    return m.group() if m else ""


def main():
    rows_in = []
    with D1_IN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            msgs = r.get("messages") or []
            if not msgs:
                continue
            user_prompt = msgs[0].get("content", "").strip()
            if not user_prompt:
                continue
            rows_in.append({"id": r.get("id"), "prompt": user_prompt})
    print(f"[d1] loaded {len(rows_in)} D1 教材条目", flush=True)

    random.seed(42)
    out_rows = []
    used_hash = set()
    for i, r in enumerate(rows_in):
        fangji = extract_fangji_name(r["prompt"])
        for vi, tmpl in enumerate(VARIANTS):
            if fangji:
                new_prompt = tmpl.format(prompt=r["prompt"], fangji=fangji)
            else:
                # 无方剂名的条目只用原 prompt(第一个变体)
                if vi != 0:
                    continue
                new_prompt = r["prompt"]
            h = hash(new_prompt)
            if h in used_hash:
                continue
            used_hash.add(h)
            out_rows.append({
                "id": f"d1_seed_{i:04d}_{vi}",
                "prompt": new_prompt,
                "source": "medical_v4_D1_textbook",
                "category": "tcm",
                "gt_answer": None,
                "meta": {"fangji": fangji, "d1_id": r["id"], "variant": vi},
            })

    random.shuffle(out_rows)
    # 目标 1000, 有多截多
    out_rows = out_rows[:1000]
    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    var_dist = Counter(r["meta"]["variant"] for r in out_rows)
    print(f"[d1] wrote {len(out_rows)} → {OUT}", flush=True)
    print(f"[d1] variant dist: {dict(var_dist)}", flush=True)
    print(f"[d1] uniq fangji: {len({r['meta']['fangji'] for r in out_rows if r['meta']['fangji']})}", flush=True)


if __name__ == "__main__":
    main()
