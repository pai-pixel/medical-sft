"""A 数据清洗第一步: 统计报告
读 medical_dpo/ 4 个 chosen jsonl, 统计:
1. 各文件计数
2. prompt 去重后剩余
3. 回避型回答 (正则规则) 占比
4. 中英混杂占比
5. 长度分布

输出 sample_report.json + 抽样 50 条疑似回避型给用户人工审。
不写最终训练集, 只出统计 + 抽样, 等用户拍板阈值后再走过滤。
"""
import json, hashlib, re, random
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)

ROOT = Path("/mnt/data/huangjiawei/datasets_local/medical_dpo")
OUT_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "chosen_m2_18k_clean.jsonl",
    "chosen_opus_25k_clean.jsonl",
    "chosen_taskd_m2_clean.jsonl",
    "chosen_rerun_bad_4307.jsonl",
]

# 回避型特征关键词 (出现在回答开头 300 字内)
EVADE_PATTERNS = [
    r"必须\s*(?:立即|首先|强调).*?(?:就医|医生|医院|急诊|120)",
    r"建议您?\s*(?:立即|尽快)\s*(?:就医|去医院|看医生|联系医生)",
    r"请务必\s*(?:立即|尽快).*?(?:就医|医生|医院|急诊)",
    r"^[^。!]*请咨询专业医[生师]",
    r"^[^。!]*无法替代医[生师]",
    r"^[^。!]*仅供参考",
]
EVADE_RE = re.compile("|".join(EVADE_PATTERNS))

# 中英混杂特征 (回答中英文字符占比 > 15%)
EN_RE = re.compile(r"[a-zA-Z]")
CN_RE = re.compile(r"[一-鿿]")

# 短回答阈值
SHORT_THRESHOLD = 200


def english_ratio(text: str) -> float:
    en = len(EN_RE.findall(text))
    cn = len(CN_RE.findall(text))
    return en / max(en + cn, 1)


def is_evasive(answer: str) -> bool:
    head = answer[:300]
    return bool(EVADE_RE.search(head))


def main():
    seen_prompts = {}  # prompt SHA → 出现次数
    all_items = []  # (file, item)
    stats = defaultdict(int)
    by_file = defaultdict(lambda: defaultdict(int))
    length_buckets = defaultdict(int)
    evasive_samples = []  # 给用户审

    for fn in FILES:
        fp = ROOT / fn
        if not fp.exists():
            print(f"[skip] {fn} not found", flush=True)
            continue
        print(f"[read] {fn}", flush=True)
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                x = json.loads(line)
                # 兼容字段: chosen 或 answer
                ans = x.get("chosen") or x.get("answer") or ""
                prompt = x.get("prompt") or ""
                if not (prompt and ans):
                    by_file[fn]["empty_field"] += 1
                    continue

                p_sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
                seen_prompts.setdefault(p_sha, 0)
                seen_prompts[p_sha] += 1

                evade = is_evasive(ans)
                en_ratio = english_ratio(ans)
                short = len(ans) < SHORT_THRESHOLD
                mixed = en_ratio > 0.15

                stats["total"] += 1
                stats["evasive"] += int(evade)
                stats["short"] += int(short)
                stats["mixed_lang"] += int(mixed)

                by_file[fn]["total"] += 1
                by_file[fn]["evasive"] += int(evade)
                by_file[fn]["short"] += int(short)
                by_file[fn]["mixed_lang"] += int(mixed)

                # 长度分桶
                L = len(ans)
                if L < 200: bucket = "<200"
                elif L < 500: bucket = "200-500"
                elif L < 1000: bucket = "500-1000"
                elif L < 2000: bucket = "1000-2000"
                elif L < 4000: bucket = "2000-4000"
                else: bucket = ">=4000"
                length_buckets[bucket] += 1

                # 收集疑似回避样本
                if evade and len(evasive_samples) < 50:
                    evasive_samples.append({
                        "file": fn, "prompt": prompt[:200],
                        "answer_head": ans[:400],
                        "len": len(ans),
                    })

                all_items.append((fn, p_sha, x))

    # 去重: 同一 SHA 第一次出现保留
    dedup_seen = set()
    dedup_count = 0
    for fn, sha, x in all_items:
        if sha in dedup_seen:
            continue
        dedup_seen.add(sha)
        dedup_count += 1

    print()
    print("=" * 60)
    print(f"[total] {stats['total']} 条")
    print(f"[dedup_after] {dedup_count} 条 (-{stats['total'] - dedup_count} 重复)")
    print(f"[evasive] {stats['evasive']} ({stats['evasive']/max(stats['total'],1)*100:.1f}%)")
    print(f"[short<200] {stats['short']} ({stats['short']/max(stats['total'],1)*100:.1f}%)")
    print(f"[mixed_lang>15% en] {stats['mixed_lang']} ({stats['mixed_lang']/max(stats['total'],1)*100:.1f}%)")
    print()
    print("[by file]")
    for fn in FILES:
        s = by_file[fn]
        if not s: continue
        print(f"  {fn}: total={s['total']} evasive={s['evasive']} short={s['short']} mixed={s['mixed_lang']}")
    print()
    print("[length buckets]")
    for k in ["<200", "200-500", "500-1000", "1000-2000", "2000-4000", ">=4000"]:
        print(f"  {k}: {length_buckets[k]}")
    print()

    # 写报告
    report = {
        "stats": dict(stats),
        "dedup_after": dedup_count,
        "by_file": {fn: dict(by_file[fn]) for fn in FILES},
        "length_buckets": dict(length_buckets),
    }
    (OUT_DIR / "A_data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[saved] {OUT_DIR / 'A_data_report.json'}")

    # 抽样疑似回避型给用户审
    sample_path = OUT_DIR / "A_evasive_samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for s in evasive_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[saved] {sample_path} ({len(evasive_samples)} 条疑似回避样本)")


if __name__ == "__main__":
    main()
