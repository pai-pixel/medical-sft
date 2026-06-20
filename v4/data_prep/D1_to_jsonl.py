"""D1 教材方剂 csv → SFT jsonl
- 跳过 QA 状态 == 未定位 / 同名/多出处需复核
- 用 5 种 question 模板轮换 (避免单一模板)
- 严格 utf-8-sig 读 csv
- 输出 schema 同 A 数据一致 (system + messages)
"""
import csv, json, random
from pathlib import Path

random.seed(42)

ROOT = Path("C:/Users/PC/Claude脚本/medical_sft/v4/data_prep")
IN = ROOT / "D1_300_formula_textbook_extracted.csv"
OUT = ROOT / "v4_D1_chosen.jsonl"

SYSTEM_PROMPT = (
    "你是一位资深临床医师,精通中医辨证施治与现代循证医学。"
    "回答时必须给出明确的临床判断、具体方剂或药品、剂量、用法、禁忌和注意事项。"
    "信息不足时可建议补充检查,但要先给出合理的鉴别诊断和处理方向。"
    "不允许仅以'请咨询医生'回避问题。"
)

QUESTION_TEMPLATES = [
    "请说明{name}的标准组成、剂量、功用、主治。",
    "{name}的组成药物有哪些?用量、功效与主治分别是什么?",
    "请详细介绍{name}的方义、配伍特点和临床主治。",
    "{name}的出处、组成、用法、功用与主治是什么?",
    "请写出{name}的方剂组成,并说明其在中医辨证施治中的应用。",
]

# 跳过的 QA 状态
SKIP_QA = {"未定位", "同名/多出处需复核", "错配已清空"}


def build_answer(row: dict) -> str:
    """按字段顺序拼 answer, 缺字段跳过"""
    field_map = [
        ("出处", "出处"),
        ("组成", "组成"),
        ("用法", "用法"),
        ("功用", "功用"),
        ("主治", "主治"),
        ("方义", "方义"),
        ("配伍特点", "配伍特点"),
        ("注意/禁忌", "注意/禁忌"),
    ]
    parts = []
    for csv_key, label in field_map:
        v = (row.get(csv_key) or "").strip()
        if v and v not in {"-", "无", "未列", "未定位"}:
            parts.append(f"{label}: {v}")
    return "\n".join(parts)


def main():
    rows = []
    skipped = {"qa": 0, "no_compose": 0, "no_indication": 0}
    with IN.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            qa = (r.get("QA状态") or "").strip()
            if qa in SKIP_QA:
                skipped["qa"] += 1
                continue
            compose = (r.get("组成") or "").strip()
            if not compose or len(compose) < 5:
                skipped["no_compose"] += 1
                continue
            indication = (r.get("主治") or "").strip()
            if not indication:
                skipped["no_indication"] += 1
                continue
            rows.append(r)

    print(f"[input] D1 csv 300 条")
    print(f"[skipped] {skipped}")
    print(f"[kept] {len(rows)} 条")

    out_records = []
    for i, r in enumerate(rows):
        name = (r.get("方剂名") or "").strip()
        if not name:
            continue
        # 模板轮换
        q = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)].format(name=name)
        a = build_answer(r)
        if not a:
            continue
        rec = {
            "id": f"v4_D1_{i:04d}",
            "src": "D1_textbook_formula",
            "src_meta": {
                "方剂名": name,
                "出处": (r.get("出处") or "").strip(),
                "页码": (r.get("教材内章节页码") or r.get("PDF页码") or "").strip(),
                "QA状态": (r.get("QA状态") or "").strip(),
            },
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        }
        out_records.append(rec)

    # 语种校验 (钩子要求): 抽 30 条断言中文一致
    import re
    EN_RE = re.compile(r"[a-zA-Z]")
    CN_RE = re.compile(r"[一-鿿]")
    sample = random.sample(out_records, min(30, len(out_records)))
    bad = []
    for x in sample:
        for m in x["messages"]:
            text = m["content"]
            en = len(EN_RE.findall(text))
            cn = len(CN_RE.findall(text))
            if en > 0 and en / max(en + cn, 1) > 0.15:
                bad.append((x["id"], m["role"], en / (en + cn)))
    if bad:
        print(f"⚠ 语种异常 {len(bad)} 条: {bad[:3]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[saved] {len(out_records)} 条 → {OUT}")
    print(f"\n[抽样 1 条]")
    print(json.dumps(out_records[0], ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()
