from __future__ import annotations

import csv
import json
from pathlib import Path


P0_FILE = Path("D2_P0_80_western_meds_conservative.csv")
D2_400_FILE = Path("D2_400_western_meds_conservative.csv")
OUTPUT = Path("D2_400_western_meds_conservative_merged.csv")
QA_CSV = Path("D2_400_western_meds_conservative_merged_QA.csv")
SUMMARY_JSON = Path("D2_400_western_meds_conservative_merged_summary.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))


def main() -> None:
    p0_rows = {row["药品通用名"]: row for row in read_csv(P0_FILE)}
    rows = read_csv(D2_400_FILE)
    merged_count = 0
    for row in rows:
        if row.get("优先级") != "P0":
            continue
        p0 = p0_rows.get(row["药品通用名"])
        if not p0:
            continue
        original_source = row.get("来源", "")
        for field in ["别名/商品名", "适应症", "规格", "成人剂量", "特殊人群剂量", "用法", "禁忌", "不良反应", "注意事项"]:
            if p0.get(field):
                row[field] = p0[field]
        if p0.get("来源"):
            row["来源"] = original_source + "；" + p0["来源"] if original_source else p0["来源"]
        row["QA状态"] = p0.get("QA状态", row["QA状态"])
        row["QA问题"] = (
            p0.get("QA问题", "")
            + "；已合并早前P0说明书预填；39药品通为非官方来源，必须用NMPA/CDE/原厂说明书或《临床用药须知》二核后方可入训。"
        ).strip("；")
        merged_count += 1

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    qa_fields = ["序号", "优先级", "类别", "药品通用名", "QA状态", "QA问题", "来源", "NEML编号"]
    with QA_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=qa_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in qa_fields})

    summary = {
        "total": len(rows),
        "p0": sum(row["优先级"] == "P0" for row in rows),
        "p1": sum(row["优先级"] == "P1" for row in rows),
        "merged_p0_rows": merged_count,
        "nonempty_adult_dose": sum(bool(row.get("成人剂量")) for row in rows),
        "qa_status_counts": {},
        "output_csv": str(OUTPUT),
        "qa_csv": str(QA_CSV),
    }
    for row in rows:
        summary["qa_status_counts"][row["QA状态"]] = summary["qa_status_counts"].get(row["QA状态"], 0) + 1
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
