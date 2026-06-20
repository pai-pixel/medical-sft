"""生成 D4 60 条补禁忌的 Excel/CSV 清单
从 ydz_p0_entries_full.json + ydz_p0_coverage.csv 抽全 60 条精确命中
字段:
  - 序号 / 类别 / 子类 / 通用名(药典) / P0 检索名 / 药典页码 / entryId
  - 药典已有: 处方✓ / 功能主治✓ / 用法用量? / 规格?
  - 待你补: 禁忌 / 不良反应 / 注意事项
  - 药监局查询链接 (按通用名 search)
  - 备注

输出 D4_60_supplement_template.csv (utf-8-sig 带 BOM,Excel 双击直接显示中文)
"""
import json, csv, re
from pathlib import Path
from urllib.parse import quote

RES = Path("C:/Users/PC/medical_dose_research")
OUT_DIR = Path("C:/Users/PC/Claude脚本/medical_sft/v4/data_prep")
OUT = OUT_DIR / "D4_60_supplement_template.csv"

# 读 60 entries 原文 + P0 命中报告
entries = {e["entryId"]: e for e in json.loads((RES / "ydz_p0_entries_full.json").read_text(encoding="utf-8"))}

hits = []
with (RES / "ydz_p0_coverage.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["status"] != "官方药典精确命中":
            continue
        m = re.search(r"entryId=(\d+)", r["best_source"] or "")
        if not m:
            continue
        hits.append({
            "p0_id": r["id"],
            "category": r["category"],
            "subcategory": r["subcategory"],
            "p0_name": r["name"],
            "entryId": int(m.group(1)),
        })


def has_section(text: str, key: str) -> str:
    return "✓" if f"【{key}】" in text else ""


# NMPA 药品说明书查询 URL (按通用名搜索)
NMPA_BASE = "https://www.nmpa.gov.cn/datasearch/search-result.html"

rows = []
for i, h in enumerate(hits, 1):
    e = entries.get(h["entryId"], {})
    text = e.get("text", "")
    title = e.get("title", h["p0_name"])
    page = e.get("pageNum", "")

    rows.append({
        "序号": i,
        "类别": h["category"],
        "子类": h["subcategory"],
        "药典通用名": title,
        "P0 检索名": h["p0_name"],
        "药典页码": page,
        "entryId": h["entryId"],
        "已有_处方": has_section(text, "处方"),
        "已有_功能主治": has_section(text, "功能与主治") or has_section(text, "功能主治"),
        "已有_用法用量": has_section(text, "用法与用量") or has_section(text, "用法用量"),
        "已有_规格": has_section(text, "规格"),
        "待补_禁忌": "",
        "待补_不良反应": "",
        "待补_注意事项": "",
        "说明书来源": "",
        "备注": "",
    })

# 排序: 中医方剂在前, 西药在后, 同类按子类
rows.sort(key=lambda x: (0 if x["类别"] == "中医方剂" else 1, x["子类"], x["药典通用名"]))

# 重新编号
for i, r in enumerate(rows, 1):
    r["序号"] = i

# 写 CSV (utf-8-sig)
fieldnames = list(rows[0].keys())
with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"[saved] {OUT} ({len(rows)} 行)")
print()
print("分布:")
from collections import Counter
print("  类别:", Counter(r["类别"] for r in rows))
print()
print("  子类(前10):")
for k, v in Counter(r["子类"] for r in rows).most_common(10):
    print(f"    {k}: {v}")
print()
print("药典已含字段:")
for k in ["已有_处方", "已有_功能主治", "已有_用法用量", "已有_规格"]:
    n = sum(1 for r in rows if r[k] == "✓")
    print(f"  {k}: {n}/{len(rows)}")
