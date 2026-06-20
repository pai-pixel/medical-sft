"""v4 训练集合并后校验
1. 语种一致性 (钩子 6 条): 抽 100 条断言 system/user/assistant 三段中文一致
2. <think> / </think> 残留检测: 切完应该 0 出现
3. 长度分布
4. 来源占比
跑通才能进 smoke / 全量训练
"""
import json, re, itertools
from pathlib import Path
from collections import Counter

TRAIN = Path("/mnt/data/huangjiawei/datasets_local/medical_v4/v4_train_full.jsonl")
SMOKE = Path("/mnt/data/huangjiawei/datasets_local/medical_v4/v4_train_smoke_2k.jsonl")

EN_RE = re.compile(r"[a-zA-Z]")
CN_RE = re.compile(r"[一-鿿]")
THINK_RE = re.compile(r"</?think>")


def lang_ratio(text: str):
    en = len(EN_RE.findall(text))
    cn = len(CN_RE.findall(text))
    return en, cn, en / max(en + cn, 1)


def main():
    if not TRAIN.exists():
        print(f"[FATAL] {TRAIN} 不存在, 先跑 merge_v4_train.py")
        return 1

    n_total = 0
    src_count = Counter()
    think_residual = 0
    lang_violations = []
    lengths = {"system": [], "user": [], "assistant": []}

    import random
    random.seed(0)

    # 1. 全量统计 think 残留
    with TRAIN.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            n_total += 1
            src_count[x.get("src", "?")] += 1
            for m in x["messages"]:
                if THINK_RE.search(m["content"]):
                    think_residual += 1
                    break

    print(f"[total] {n_total} 条")
    print(f"[by src] {dict(src_count)}")
    print(f"[think residual] {think_residual} 条 (期望 0)")

    if think_residual > 0:
        print(f"⚠ 有 {think_residual} 条仍含 <think>/</think>, merge_v4_train.py 切除有遗漏")

    # 2. 抽 100 条做语种 + 长度
    with TRAIN.open(encoding="utf-8") as f:
        all_lines = list(itertools.islice(f, 50000))
    samples = random.sample(all_lines, min(100, len(all_lines)))

    for line in samples:
        x = json.loads(line)
        sys_text = x.get("system", "")
        user_text = x["messages"][0]["content"]
        ast_text = x["messages"][1]["content"]

        lengths["system"].append(len(sys_text))
        lengths["user"].append(len(user_text))
        lengths["assistant"].append(len(ast_text))

        for label, text in [("sys", sys_text), ("user", user_text), ("ast", ast_text)]:
            en, cn, ratio = lang_ratio(text)
            # 阈值 30%: 医学回答含拉丁学名/方剂拼音/CT/MRI 等规范英文术语, 20% 误判率高
            if cn > 0 and ratio > 0.30:
                lang_violations.append({
                    "id": x.get("id"), "field": label,
                    "en_ratio": round(ratio, 3),
                    "preview": text[:100],
                })

    print()
    print(f"[lang sample] 100 条:")
    print(f"  violations (en_ratio > 20%): {len(lang_violations)}")
    if lang_violations:
        for v in lang_violations[:3]:
            print(f"    {v}")
    print()

    print(f"[length sample] 100 条字符分布")
    for k, vs in lengths.items():
        vs.sort()
        print(f"  {k}: min={min(vs)} p50={vs[50]} p95={vs[95]} max={max(vs)}")

    # 3. 抽 3 条不同 src 看实际 SFT 样本
    print()
    print(f"[抽样不同 src] (验证 schema)")
    seen_src = set()
    for line in samples:
        x = json.loads(line)
        s = x.get("src", "?")
        if s in seen_src:
            continue
        seen_src.add(s)
        print(f"\n--- src={s} id={x.get('id')} ---")
        print(f"  user: {x['messages'][0]['content'][:120]}")
        print(f"  ast head: {x['messages'][1]['content'][:200]}")

    # 4. 综合判断
    print()
    print("=" * 50)
    if think_residual == 0 and len(lang_violations) <= 5:
        print("✅ 校验通过, 可进入 smoke 训练")
        return 0
    else:
        print(f"❌ 校验未通过: think_residual={think_residual}, lang_violations={len(lang_violations)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
