"""Phase 4: 过滤 v6 answer,生成干净 SFT 格式 jsonl。

输入: /mnt/data/huangjiawei/datasets_local/medical_v6/answers/opus_all.jsonl.staging (19660)
输出: /mnt/data/huangjiawei/datasets_local/medical_v6/v6_new_clean.jsonl

过滤规则(参考 v5 merge_v5_train.py + 钩子铁律 5/6):
- finish_reason=length 丢
- final 段 <150 字丢
- final 仍 evasive 丢
- 全段 CN 汉字占比 <35% 丢 (thinking 段可以短,但整体应中文占主导)
- final 段 CN 占比 <45% 丢 (更严格)
- MCQ 类必须含 "答案:" 或 "答案是" 标识

输出 SFT schema:
  {"id","system","messages":[user,assistant]}
统一 SYSTEM_PROMPT (跟 v5 一致)
"""
import json, re, hashlib, sys
from pathlib import Path
from collections import Counter, defaultdict

V6_DIR = Path("/mnt/data/huangjiawei/datasets_local/medical_v6")
INP = Path("/tmp/opus_all.jsonl")  # 已 cp 到 /tmp,防 virtiofs 慢读
OUT = V6_DIR / "v6_new_clean.jsonl"

SYSTEM_PROMPT = (
    "你是一位资深临床医师,精通中医辨证施治与现代循证医学。"
    "回答时必须给出明确的临床判断、具体方剂或药品、剂量、用法、禁忌和注意事项。"
    "信息不足时可建议补充检查,但要先给出合理的鉴别诊断和处理方向。"
    "不允许仅以'请咨询医生'回避问题。"
)

THINK_END_RE = re.compile(r"</think>\s*", re.DOTALL)
CN_RE = re.compile(r"[一-鿿]")

EVADE_PATTERNS = [
    r"^[^。!]{0,30}?请务必\s*(?:立即|尽快)\s*(?:就医|去医院|看医生|联系医生|咨询医生)",
    r"^[^。!]{0,30}?建议您?\s*(?:立即|尽快)\s*(?:就医|去医院|看医生|咨询医生|联系医生)",
    r"^[^。!]{0,30}?请咨询专业医[生师]",
    r"^[^。!]{0,30}?无法替代医[生师]",
    r"^[^。!]{0,30}?仅供参考,不能替代",
]
EVADE_RE = re.compile("|".join(EVADE_PATTERNS))

MCQ_ANSWER_RE = re.compile(r"答案[::是为]|Answer[:：]")


def split_thinking(text: str):
    m = THINK_END_RE.search(text)
    if not m:
        return "", text
    return text[:m.end()], text[m.end():]


def cn_ratio(s: str) -> float:
    if not s: return 0.0
    return sum(1 for c in s if CN_RE.match(c)) / max(len(s), 1)


def is_evasive(final_head: str) -> bool:
    return bool(EVADE_RE.search(final_head[:300]))


def main():
    stats = Counter()
    kept_by_cat = defaultdict(int)
    dropped_by_reason = Counter()
    seen_prompt_hash = set()
    kept = []

    with INP.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                stats["json_error"] += 1
                continue
            stats["total"] += 1
            ans = r.get("answer", "")
            cat = r.get("category", "")
            prompt = r.get("prompt", "")

            if r.get("finish_reason") == "length":
                dropped_by_reason["truncated"] += 1
                continue
            if not ans or not prompt:
                dropped_by_reason["empty"] += 1
                continue

            thinking, final = split_thinking(ans)
            if not final.strip():
                dropped_by_reason["no_close_think"] += 1
                continue
            if len(final.strip()) < 150:
                dropped_by_reason["final_too_short"] += 1
                continue
            if is_evasive(final):
                dropped_by_reason["final_evasive"] += 1
                continue

            # 语种校验(钩子铁律 5)
            all_cn = cn_ratio(ans)
            final_cn = cn_ratio(final)
            if all_cn < 0.35:
                dropped_by_reason["cn_ratio_low_all"] += 1
                continue
            if final_cn < 0.45:
                dropped_by_reason["cn_ratio_low_final"] += 1
                continue

            # MCQ 必须含答案标识
            if cat == "mcq" and not MCQ_ANSWER_RE.search(final):
                dropped_by_reason["mcq_no_answer"] += 1
                continue

            # 去重(prompt hash)
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
            if h in seen_prompt_hash:
                dropped_by_reason["dup_prompt"] += 1
                continue
            seen_prompt_hash.add(h)

            # 写盘: 保留完整 answer 含 thinking(跟 v5 一致)
            rec = {
                "id": f"v6_{cat}_{kept_by_cat[cat]:05d}",
                "src": f"v6_opus_{r.get('teacher', 'opus')}",
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ans},  # 整段保留 <think>
                ],
            }
            kept.append(rec)
            kept_by_cat[cat] += 1

    with OUT.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"total in: {stats['total']}")
    print(f"kept: {len(kept)} ({len(kept)/stats['total']*100:.1f}%)")
    print(f"kept by cat: {dict(kept_by_cat)}")
    print(f"dropped by reason: {dict(dropped_by_reason)}")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
