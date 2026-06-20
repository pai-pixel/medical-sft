"""D 数据 60 条精确命中转 SFT schema
读 ydz_selected_entries.json (含完整原文) + ydz_p0_coverage.csv (60 精确命中清单)
输出 v4 SFT 格式 jsonl

关键约定:
- 药典条目 = 中成药制剂, 不是教材煎汤方剂
- answer 主体: 用法与用量 + 功能与主治 + 规格 (临床实用)
- 处方部分: 单独段落标注 "以下为中成药制剂处方 (制成中成药每批的原料配比), 非临床煎汤剂量"
- 缺禁忌: 不补造, 留空标记 source 可追溯, 后续 D 教材方剂部分人工补
"""
import json, re, csv, sys
from pathlib import Path

ROOT = Path("C:/Users/PC/medical_dose_research")
OUT_DIR = Path("C:/Users/PC/Claude脚本/medical_sft/v4/data_prep")
OUT = OUT_DIR / "D_official_60.jsonl"

# 读 entries (按 entryId 索引)
entries = json.loads((ROOT / "ydz_p0_entries_full.json").read_text(encoding="utf-8"))
by_id = {e["entryId"]: e for e in entries}
print(f"[entries loaded] {len(entries)} 条 official entries")

# 读 P0 命中清单 - 找出精确命中
hits = []
with (ROOT / "ydz_p0_coverage.csv").open(encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["status"] != "官方药典精确命中":
            continue
        # 从 best_source 抠 entryId
        m = re.search(r"entryId=(\d+)", r["best_source"] or "")
        if not m:
            continue
        eid = int(m.group(1))
        hits.append({
            "id": r["id"],
            "category": r["category"],
            "subcategory": r["subcategory"],
            "name": r["name"],
            "entryId": eid,
            "best_title": r["best_title"],
            "best_source": r["best_source"],
        })
print(f"[hits parsed] {len(hits)} 精确命中条目, 唯一 entryId {len({h['entryId'] for h in hits})} 个")

# 检查 entries 覆盖率
missing_entries = [h for h in hits if h["entryId"] not in by_id]
covered = [h for h in hits if h["entryId"] in by_id]
print(f"[coverage] {len(covered)}/{len(hits)} 有原文, {len(missing_entries)} 缺原文 (需 fetch_ydz.py 补抓)")

if missing_entries:
    print("\n缺原文的前 5 条:")
    for h in missing_entries[:5]:
        print(f"  - {h['id']} {h['name']} entryId={h['entryId']}")


def parse_sections(text: str) -> dict:
    """从 entry text 解析【XX】段落"""
    sections = {}
    pattern = r"【([^】]+)】\s*\n?([^【]*)"
    for m in re.finditer(pattern, text):
        key = m.group(1).strip()
        val = m.group(2).strip()
        sections[key] = val
    return sections


def is_zhongchengyao(name: str, entry_title: str, sections: dict) -> bool:
    """判断是否为中成药条目 (有【处方】+【制法】+ 名字含 丸/片/胶囊/颗粒/口服液/酊/膏/散等剂型, 或处方剂量是 g 大单位)"""
    if any(k in entry_title for k in ["丸", "片", "胶囊", "颗粒", "口服液", "酊", "膏", "散", "饮", "注射液", "栓"]):
        return True
    rx = sections.get("处方", "")
    # 处方剂量含 g 大单位 (>= 50g 单味药) 是制剂特征
    if re.search(r"\d{2,}g", rx):
        return True
    return False


def build_answer(entry: dict, sections: dict) -> str:
    """组装 v4 训练用 answer
    主体: 用法用量 + 功能主治 + 规格 (临床实用)
    辅助: 处方 (标注为中成药生产处方, 非煎汤剂量)
    """
    parts = []

    if "功能与主治" in sections:
        parts.append(f"功能与主治: {sections['功能与主治']}")
    elif "功能主治" in sections:
        parts.append(f"功能主治: {sections['功能主治']}")

    if "用法与用量" in sections:
        parts.append(f"用法用量: {sections['用法与用量']}")
    elif "用法用量" in sections:
        parts.append(f"用法用量: {sections['用法用量']}")

    if "规格" in sections:
        parts.append(f"规格: {sections['规格']}")

    # 处方单独段落, 加 disclaimer
    if "处方" in sections:
        rx = sections["处方"].strip()
        parts.append(
            f"中成药制剂处方(每批生产原料配比, 非临床煎汤剂量): {rx}"
        )

    if "贮藏" in sections:
        parts.append(f"贮藏: {sections['贮藏']}")

    return "\n".join(parts)


# 转换
out_records = []
skipped = []
for h in covered:
    entry = by_id[h["entryId"]]
    sections = parse_sections(entry.get("text", ""))

    if "用法与用量" not in sections and "用法用量" not in sections:
        skipped.append((h, "no_dosage"))
        continue
    if "功能与主治" not in sections and "功能主治" not in sections:
        skipped.append((h, "no_indication"))
        continue

    is_zcy = is_zhongchengyao(h["name"], entry["title"], sections)
    if not is_zcy:
        # 不是中成药也保留, 但标 type 不同
        item_type = "饮片/单味药"
    else:
        item_type = "中成药制剂"

    # question 模板 - 强调"中成药制剂"
    if h["category"] == "中医方剂":
        question = (
            f"{entry['title']}(中成药制剂)的标准功能主治、用法用量、规格、处方组成是什么?"
            if is_zcy else
            f"{entry['title']}的标准功能主治、用法用量是什么?"
        )
    else:
        question = (
            f"{entry['title']}的标准适应症、用法用量、规格是什么?"
        )

    answer = build_answer(entry, sections)

    record = {
        "id": f"D_OFFICIAL_{h['entryId']:04d}",
        "source": "v4_D_official",
        "category": h["category"],
        "subcategory": h["subcategory"],
        "type": item_type,
        "name_p0": h["name"],
        "name_pharmacopeia": entry["title"],
        "entryId": h["entryId"],
        "page": entry.get("pageNum"),
        "source_ref": h["best_source"],
        "question": question,
        "answer": answer,
        "verification_status": "verified_pharmacopeia_2020",
        "missing_fields": [
            k for k in ["禁忌", "不良反应", "注意"]
            if k not in sections
        ],
    }
    out_records.append(record)

# 写
OUT_DIR.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for r in out_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n[out] {len(out_records)} 条写入 {OUT}")
print(f"[skipped] {len(skipped)} 条 (字段不全)")
if skipped:
    print("跳过的前 5 条:")
    for h, reason in skipped[:5]:
        print(f"  - {h['id']} {h['name']} reason={reason}")

# 统计 missing_fields
from collections import Counter
mf_count = Counter()
for r in out_records:
    for k in r["missing_fields"]:
        mf_count[k] += 1
print(f"\n[missing fields 统计] (这些药典本身没有, 需要后续从说明书补)")
for k, v in mf_count.most_common():
    print(f"  {k}: {v}/{len(out_records)} ({v/len(out_records)*100:.0f}%)")

# 抽样输出
print(f"\n[抽样 1 条]")
print(json.dumps(out_records[0], ensure_ascii=False, indent=2)[:1500])
