import csv
from pathlib import Path


INPUT = Path("D2_P0_80_western_meds_39ypk_draft.csv")
OUTPUT = Path("D2_P0_80_western_meds_conservative.csv")

BAD_ROWS = {
    "45": "吗啡匹配为检测试纸，非药品说明书",
    "75": "口服补液盐III匹配为口服补液盐散(Ⅱ)，剂型版本错误",
}


def main():
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8-sig", newline="")))
    fieldnames = list(rows[0].keys())
    for row in rows:
        if row["序号"] in BAD_ROWS:
            for field in ["别名/商品名", "适应症", "规格", "成人剂量", "特殊人群剂量", "用法", "禁忌", "不良反应", "注意事项", "来源"]:
                row[field] = ""
            row["QA状态"] = "错配已清空"
            row["QA问题"] = BAD_ROWS[row["序号"]] + "；需CDE/NMPA/原厂说明书重新检索"
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
