"""v4 训练集合并 v2 — 修两个 bug
修复 1: filter_c 加切 thinking 段 (re.sub <think>...</think>)
修复 2: merge 顺序 C → D1 → A (C 优先入,A 跳过 C 重复 prompt)
"""
import json, re, hashlib
from pathlib import Path
from collections import Counter

ROOT = Path("/mnt/data/huangjiawei/datasets_local/medical_v4")
A_FILE = ROOT / "chosen_v4_filtered.jsonl"
C_RAW = ROOT / "v4_C_chosen.jsonl"
C_CLEAN = ROOT / "v4_C_chosen_clean.jsonl"
D1_FILE = ROOT / "v4_D1_chosen.jsonl"
TRAIN = ROOT / "v4_train_full.jsonl"
SMOKE = ROOT / "v4_train_smoke_2k.jsonl"

SYSTEM_PROMPT = (
    "你是一位资深临床医师,精通中医辨证施治与现代循证医学。"
    "回答时必须给出明确的临床判断、具体方剂或药品、剂量、用法、禁忌和注意事项。"
    "信息不足时可建议补充检查,但要先给出合理的鉴别诊断和处理方向。"
    "不允许仅以'请咨询医生'回避问题。"
)

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)  # 没闭合的 thinking 段

EVADE_PATTERNS = [
    r"必须\s*(?:立即|首先|强调).*?(?:就医|医生|医院|急诊|120)",
    r"建议您?\s*(?:立即|尽快)\s*(?:就医|去医院|看医生|联系医生)",
    r"请务必\s*(?:立即|尽快).*?(?:就医|医生|医院|急诊)",
    r"^[^。!]*请咨询专业医[生师]",
    r"^[^。!]*无法替代医[生师]",
    r"^[^。!]*仅供参考",
]
EVADE_RE = re.compile("|".join(EVADE_PATTERNS))


def strip_thinking(text: str) -> str:
    """切除 <think>...</think> 段, 兼容没闭合的(只有 <think> 没 </think>)"""
    text = THINK_RE.sub("", text)
    # 残留没闭合的 <think>...EOF
    text = THINK_OPEN_RE.sub("", text)
    return text.strip()


def is_evasive(text: str) -> bool:
    return bool(EVADE_RE.search(text[:300]))


def filter_c():
    """过滤 C, 切 thinking, 输出 v4_C_chosen_clean.jsonl"""
    if not C_RAW.exists():
        print(f"[c] {C_RAW} not found, skip")
        return 0
    stats = Counter()
    out = []
    with C_RAW.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            stats["total"] += 1
            raw_ans = r.get("new_answer", "")

            if r.get("finish_reason") == "length":
                stats["truncated"] += 1
                continue

            # 切 thinking 段
            cleaned = strip_thinking(raw_ans)

            if len(cleaned) < 200:
                stats["short_after_strip"] += 1
                continue

            if is_evasive(cleaned):
                stats["still_evasive"] += 1
                continue

            r["_cleaned_answer"] = cleaned
            out.append(r)
            stats["kept"] += 1

    print(f"[c] {dict(stats)}")
    with C_CLEAN.open("w", encoding="utf-8") as f:
        for i, r in enumerate(out):
            rec = {
                "id": f"v4_C_{i:05d}",
                "src": "C_m2_rewritten",
                "src_meta": {
                    "src_id": r.get("src_id"),
                    "raw_tok_count": r.get("tok_count"),
                    "cleaned_chars": len(r["_cleaned_answer"]),
                },
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["_cleaned_answer"]},
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(out)


def merge_train():
    """合并 C → D1 → A (C 优先, A 跳过 C 重复 prompt)
    统一 schema: 只保留 id + system + messages, 丢其他字段防止 datasets 加载冲突
    """
    seen_sha = set()
    counts = Counter()
    dups = Counter()

    with TRAIN.open("w", encoding="utf-8") as fout:
        # === 顺序: C 优先, D1 次, A 最后 ===
        for label, path in [("C", C_CLEAN), ("D1", D1_FILE), ("A", A_FILE)]:
            if not path.exists():
                print(f"[merge] {path} 不存在, skip")
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    prompt = r["messages"][0]["content"]
                    sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
                    if sha in seen_sha:
                        dups[label] += 1
                        continue
                    seen_sha.add(sha)
                    # 统一 schema: 只保留训练必须的 3 个字段
                    rec = {
                        "id": r.get("id", f"{label}_{counts[label]}"),
                        "system": r["system"],
                        "messages": r["messages"],
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts[label] += 1

    total = sum(counts.values())
    print(f"\n[merge] kept by source: {dict(counts)}")
    print(f"[merge] dropped (dup with earlier source): {dict(dups)}")
    print(f"[merge] total: {total}")

    # 抽 2k smoke 子集
    import random
    random.seed(42)
    with TRAIN.open(encoding="utf-8") as f:
        all_lines = f.readlines()
    smoke = random.sample(all_lines, min(2000, len(all_lines)))
    with SMOKE.open("w", encoding="utf-8") as f:
        f.writelines(smoke)
    print(f"[smoke] {len(smoke)} 条 → {SMOKE}")
    return total


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: filter C + strip thinking")
    print("=" * 60)
    nc = filter_c()
    print()
    print("=" * 60)
    print("Step 2: merge C → D1 → A (dedup)")
    print("=" * 60)
    n = merge_train()
    print()
    print(f"[done] v4 训练集 {n} 条 → {TRAIN}")
