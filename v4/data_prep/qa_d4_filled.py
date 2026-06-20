import csv
import sys
from collections import Counter
from pathlib import Path


INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D4_60_supplement_filled_39ypk.csv")
QA_OUTPUT = INPUT.with_name(INPUT.stem + "_QA.csv")
USABLE_OUTPUT = INPUT.with_name(INPUT.stem + "_nonofficial_usable.csv")


SUSPICIOUS_TERMS = ["可是", "斟酌", "其咨询", "严惩病例", "职出现", "肝重肾功能", "阿司林"]


def source_title(source: str):
    parts = source.split("；")
    return parts[1] if len(parts) > 1 else ""


def main():
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8-sig", newline="")))
    qa_rows = []
    usable_rows = []
    for row in rows:
        issues = []
        if not row.get("待补_禁忌"):
            issues.append("禁忌空")
        if not row.get("待补_不良反应"):
            issues.append("不良反应空")
        if not row.get("待补_注意事项"):
            issues.append("注意事项空")
        all_text = " ".join(
            [
                row.get("待补_禁忌", ""),
                row.get("待补_不良反应", ""),
                row.get("待补_注意事项", ""),
                row.get("说明书来源", ""),
            ]
        )
        for term in SUSPICIOUS_TERMS:
            if term in all_text:
                issues.append(f"疑似文本错误:{term}")
        title = source_title(row.get("说明书来源", ""))
        name = row.get("药典通用名", "")
        query = row.get("P0 检索名", "")
        if title and name not in title and query not in title and title not in name:
            issues.append(f"候选标题需核:{title}")
        if "非NMPA官网源" in row.get("备注", ""):
            issues.append("非NMPA官网源")
        if not row.get("说明书来源"):
            issues.append("无说明书来源")

        qa_status = "可作为非官方初筛"
        if any(issue.endswith("空") or issue == "无说明书来源" for issue in issues):
            qa_status = "未完成"
        elif any(issue.startswith("疑似文本错误") or issue.startswith("候选标题需核") for issue in issues):
            qa_status = "需人工复核"
        elif "非NMPA官网源" in issues:
            qa_status = "需NMPA/原厂二核"

        qa_row = dict(row)
        qa_row["QA状态"] = qa_status
        qa_row["QA问题"] = "；".join(issues)
        qa_rows.append(qa_row)
        if qa_status in {"可作为非官方初筛", "需NMPA/原厂二核"}:
            usable_rows.append(qa_row)

    fieldnames = list(rows[0].keys()) + ["QA状态", "QA问题"]
    with QA_OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qa_rows)
    with USABLE_OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(usable_rows)

    counts = Counter(row["QA状态"] for row in qa_rows)
    for status, count in counts.items():
        print(f"{status}: {count}")
    print(f"QA: {QA_OUTPUT}")
    print(f"usable_nonofficial: {USABLE_OUTPUT}")


if __name__ == "__main__":
    main()
