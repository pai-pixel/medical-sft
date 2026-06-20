import csv
from pathlib import Path


INPUT = Path("D4_60_supplement_filled_39ypk.csv")
OUTPUT = Path("D4_60_supplement_filled_conservative.csv")


MANUAL_FIXES = {
    "8": {
        "待补_禁忌": "对本品及牛乳过敏者禁用。",
        "备注_append": "39药品通疑似录入错字“可是牛乳”，已按语义修正为“及牛乳”；仍需原厂/NMPA二核",
    },
    "9": {
        "待补_禁忌": "",
        "备注_append": "39药品通禁忌字段为“斟酌”，疑似错误，已清空待核",
    },
    "23": {
        "待补_注意事项_replace": ("血镁浓度职出现", "血镁浓度；出现"),
        "备注_append": "39药品通疑似录入错字“职出现”，已按语义加分号；仍需原厂/NMPA二核",
    },
    "47": {
        "待补_禁忌": "",
        "待补_不良反应": "",
        "待补_注意事项": "",
        "说明书来源": "",
        "备注_append": "39药品通候选错配为依降钙素注射液，已清空待重新检索",
    },
    "51": {
        "待补_禁忌_replace": ("阿司林", "阿司匹林"),
        "备注_append": "39药品通疑似录入错字“阿司林”，已修正为“阿司匹林”；仍需原厂/NMPA二核",
    },
}


def append_note(row, note):
    row["备注"] = (row.get("备注", "") + "；" + note).strip("；")


def main():
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())

    for row in rows:
        fix = MANUAL_FIXES.get(row["序号"])
        if not fix:
            continue
        for key, value in fix.items():
            if key == "备注_append":
                append_note(row, value)
            elif key.endswith("_replace"):
                field = key[: -len("_replace")]
                old, new = value
                row[field] = row.get(field, "").replace(old, new)
            else:
                row[key] = value

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
