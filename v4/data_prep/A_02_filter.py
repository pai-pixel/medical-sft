"""A 数据清洗第二步: 执行过滤
策略 (基于第一步报告 + 50 样本审查):
  - 删 short < 200 字 (SFT 价值低)
  - 删 mixed_lang > 15% en (单语种铁律)
  - 保留 evasive (多数是 prompt 信息不足导致的合理临床建议, 误判率高)
  - 删 prompt 重复 (本批数据已 dedup, 但加保险)

输出 chosen_v4_filtered.jsonl, schema:
  {"id": ..., "domain": ..., "system": ..., "messages": [{"role":"user", ...}, {"role":"assistant",...}]}
"""
import json, hashlib, re
from pathlib import Path
from collections import defaultdict

ROOT = Path("/mnt/data/huangjiawei/datasets_local/medical_dpo")
OUT_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "chosen_v4_filtered.jsonl"

FILES = [
    "chosen_m2_18k_clean.jsonl",
    "chosen_opus_25k_clean.jsonl",
    "chosen_taskd_m2_clean.jsonl",
    "chosen_rerun_bad_4307.jsonl",
]

EN_RE = re.compile(r"[a-zA-Z]")
CN_RE = re.compile(r"[一-鿿]")
SHORT_TH = 200
MIXED_TH = 0.15

# v4 single system prompt — 不再 v2 双轨, 干净起点
SYSTEM_PROMPT = (
    "你是一位资深临床医师,精通中医辨证施治与现代循证医学。"
    "回答时必须给出明确的临床判断、具体方剂或药品、剂量、用法、禁忌和注意事项。"
    "信息不足时可建议补充检查,但要先给出合理的鉴别诊断和处理方向。"
    "不允许仅以'请咨询医生'回避问题。"
)


def english_ratio(text: str) -> float:
    en = len(EN_RE.findall(text))
    cn = len(CN_RE.findall(text))
    return en / max(en + cn, 1)


def main():
    seen_sha = set()
    kept = 0
    dropped = defaultdict(int)
    fout = open(OUT, "w", encoding="utf-8", buffering=1)

    for fn in FILES:
        fp = ROOT / fn
        if not fp.exists():
            continue
        print(f"[read] {fn}", flush=True)
        n_kept_file = 0
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                x = json.loads(line)
                ans = x.get("chosen") or x.get("answer") or ""
                prompt = x.get("prompt") or ""
                domain = x.get("domain", "")

                if not (prompt and ans):
                    dropped["empty"] += 1
                    continue

                sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
                if sha in seen_sha:
                    dropped["dedup"] += 1
                    continue
                seen_sha.add(sha)

                # 删 short
                if len(ans) < SHORT_TH:
                    dropped["short"] += 1
                    continue

                # 删 mixed
                en_r = english_ratio(ans)
                if en_r > MIXED_TH:
                    dropped["mixed"] += 1
                    continue

                # 保留 - 转 v4 schema
                rec = {
                    "id": f"v4_A_{kept:06d}",
                    "src_id": x.get("id"),
                    "src_file": fn,
                    "domain": domain,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": ans},
                    ],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                n_kept_file += 1
        print(f"  → kept {n_kept_file}", flush=True)

    fout.close()

    print(f"\n{'=' * 50}")
    print(f"[kept] {kept} 条")
    print(f"[dropped]")
    for k, v in sorted(dropped.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"\n[saved] {OUT}")

    # 写 stats
    stats = {"kept": kept, "dropped": dict(dropped)}
    (OUT_DIR / "A_filter_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
