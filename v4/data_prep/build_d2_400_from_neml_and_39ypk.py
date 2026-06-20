from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from pypdf import PdfReader

import fill_d2_from_39ypk as ypk


NEML_PDF = Path("NHC_NEML_2018.pdf")
NEML_TEXT_DIR = Path("NHC_NEML_2018_pages_text")
TEMPLATE = Path("D2_400_western_meds_template_from_NEML2018.csv")
DRAFT = Path("D2_400_western_meds_39ypk_draft.csv")
OUTPUT = Path("D2_400_western_meds_conservative.csv")
QA_CSV = Path("D2_400_western_meds_QA.csv")
SUMMARY_JSON = Path("D2_400_western_meds_summary.json")
RAW_DIR = Path("D2_400_39ypk_raw")


P0_NAMES = [
    "青霉素G",
    "阿莫西林",
    "阿莫西林克拉维酸钾",
    "头孢呋辛",
    "头孢曲松",
    "头孢哌酮舒巴坦",
    "阿奇霉素",
    "克拉霉素",
    "左氧氟沙星",
    "莫西沙星",
    "甲硝唑",
    "万古霉素",
    "氨氯地平",
    "硝苯地平控释片",
    "依那普利",
    "缬沙坦",
    "厄贝沙坦",
    "美托洛尔缓释片",
    "比索洛尔",
    "拉贝洛尔",
    "氢氯噻嗪",
    "呋塞米",
    "二甲双胍",
    "格列美脲",
    "阿卡波糖",
    "达格列净",
    "恩格列净",
    "利拉鲁肽",
    "甘精胰岛素",
    "门冬胰岛素",
    "阿托伐他汀",
    "瑞舒伐他汀",
    "依折麦布",
    "依洛尤单抗",
    "阿司匹林肠溶片",
    "氯吡格雷",
    "替格瑞洛",
    "华法林",
    "利伐沙班",
    "达比加群酯",
    "依诺肝素",
    "对乙酰氨基酚",
    "布洛芬",
    "塞来昔布",
    "吗啡",
    "芬太尼贴剂",
    "曲马多",
    "沙丁胺醇气雾剂",
    "布地奈德吸入剂",
    "孟鲁司特",
    "异丙托溴铵",
    "噻托溴铵",
    "奥美拉唑",
    "雷贝拉唑",
    "枸橼酸铋钾",
    "莫沙必利",
    "蒙脱石散",
    "舍曲林",
    "艾司西酞普兰",
    "文拉法辛",
    "地西泮",
    "奥氮平",
    "利培酮",
    "泼尼松",
    "甲泼尼龙",
    "地塞米松",
    "氢化可的松",
    "左甲状腺素钠",
    "甲巯咪唑",
    "丙硫氧嘧啶",
    "米非司酮",
    "米索前列醇",
    "缩宫素",
    "低分子肝素",
    "口服补液盐III",
    "蒙脱石散(小儿)",
    "布洛芬混悬液(小儿剂量)",
    "碳酸钙D3",
    "骨化三醇",
    "阿仑膦酸钠",
]


ALIASES = {
    "青霉素G": "青霉素",
    "头孢哌酮舒巴坦": "头孢哌酮钠舒巴坦钠",
    "硝苯地平控释片": "硝苯地平",
    "美托洛尔缓释片": "美托洛尔",
    "依洛尤单抗": "依洛尤单抗",
    "阿司匹林肠溶片": "阿司匹林",
    "芬太尼贴剂": "芬太尼",
    "沙丁胺醇气雾剂": "沙丁胺醇",
    "布地奈德吸入剂": "布地奈德",
    "异丙托溴铵": "异丙托溴铵",
    "噻托溴铵": "噻托溴铵",
    "低分子肝素": "低分子肝素",
    "口服补液盐III": "口服补液盐",
    "蒙脱石散(小儿)": "蒙脱石散",
    "布洛芬混悬液(小儿剂量)": "布洛芬",
    "碳酸钙D3": "碳酸钙D3",
}


CLASS_MAP = [
    ("青霉素|阿莫西林|头孢|阿奇霉素|克拉霉素|左氧氟沙星|莫西沙星|甲硝唑|万古霉素|利奈唑胺|诺氟沙星|环丙沙星|替硝唑|呋喃妥因|阿米卡星|庆大霉素|多西环素|米诺环素|氯霉素|克林霉素|磷霉素|复方磺胺甲噁唑|异烟肼|利福平|乙胺丁醇|吡嗪酰胺|氟康唑|制霉素|阿昔洛韦|奥司他韦", "抗感染"),
    ("氨氯地平|硝苯地平|依那普利|缬沙坦|厄贝沙坦|美托洛尔|比索洛尔|拉贝洛尔|氢氯噻嗪|呋塞米|硝普钠|尼群地平|卡托普利|贝那普利|氯沙坦|氨苯蝶啶|螺内酯|吲达帕胺", "心血管/降压"),
    ("二甲双胍|格列美脲|阿卡波糖|达格列净|恩格列净|利拉鲁肽|胰岛素|甲状腺|甲巯咪唑|丙硫氧嘧啶|格列本脲|格列吡嗪", "内分泌"),
    ("阿托伐他汀|瑞舒伐他汀|辛伐他汀|依折麦布|依洛尤单抗", "调脂"),
    ("阿司匹林|氯吡格雷|替格瑞洛|华法林|利伐沙班|达比加群|依诺肝素|肝素", "抗血栓"),
    ("对乙酰氨基酚|布洛芬|塞来昔布|吗啡|芬太尼|曲马多|哌替啶|双氯芬酸|吲哚美辛", "镇痛/解热"),
    ("沙丁胺醇|布地奈德|孟鲁司特|异丙托溴铵|噻托溴铵|氨茶碱|乙酰半胱氨酸|羧甲司坦", "呼吸"),
    ("奥美拉唑|雷贝拉唑|铋|莫沙必利|蒙脱石|多潘立酮|甲氧氯普胺|乳果糖|开塞露", "消化"),
    ("舍曲林|艾司西酞普兰|文拉法辛|地西泮|奥氮平|利培酮|氯硝西泮|卡马西平|丙戊酸|苯巴比妥|苯妥英|左乙拉西坦", "精神/神经"),
    ("泼尼松|甲泼尼龙|地塞米松|氢化可的松", "激素"),
    ("米非司酮|米索前列醇|缩宫素|硫酸镁|卡前列|雌二醇|黄体酮", "妇产"),
    ("口服补液盐|蒙脱石|布洛芬混悬", "儿科常用"),
    ("碳酸钙|骨化三醇|阿仑膦酸", "老年/慢病"),
]


def category_for(name: str) -> str:
    for pattern, category in CLASS_MAP:
        if re.search(pattern, name):
            return category
    return "国家基本药物目录扩展"


def clean_lines(text: str) -> str:
    return text


def extract_neml_pages() -> list[str]:
    NEML_TEXT_DIR.mkdir(exist_ok=True)
    reader = PdfReader(str(NEML_PDF))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = clean_lines(text)
        pages.append(text)
        (NEML_TEXT_DIR / f"page_{index:03d}.txt").write_text(text, "utf-8")
    return pages


def is_english_line(line: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 ,()/:.-]+", line))


def is_chinese_name_line(line: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", line)) and not bool(
        re.search(r"序号|品种名称|剂型|规格|备注|第一部分|国家基本药物目录|化学药品|生物制品|^[一二三四五六七八九十]+、|[：:]", line)
    )


def parse_neml_entries() -> list[dict[str, str]]:
    pages = extract_neml_pages()
    entries: list[dict[str, str]] = []
    for page_index, text in enumerate(pages, 1):
        if page_index < 13 or page_index > 69:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if not re.fullmatch(r"\d{1,3}", line):
                continue
            number = int(line)
            if number < 1 or number > 417:
                continue
            name_parts: list[str] = []
            english_parts: list[str] = []
            after_name = False
            for probe in lines[index + 1 : index + 10]:
                if is_english_line(probe):
                    after_name = True
                    english_parts.append(probe)
                    continue
                if not after_name and is_chinese_name_line(probe):
                    name_parts.append(probe)
                    continue
                if after_name:
                    break
            name = "".join(name_parts)
            english = " ".join(english_parts)
            if not name:
                continue
            spec_lines: list[str] = []
            collecting = False
            for probe in lines[index + 1 : index + 20]:
                if probe in name_parts or probe in english_parts:
                    collecting = True
                    continue
                if collecting:
                    if re.fullmatch(r"\d{1,3}", probe):
                        break
                    if re.search(r"序号|品种名称|剂型|规格|备注|^[一二三四五六七八九十]+、", probe):
                        break
                    spec_lines.append(probe)
            spec = "；".join(spec_lines[:8])
            entries.append(
                {
                    "neml_no": str(number),
                    "name": name,
                    "english": english,
                    "spec": spec,
                    "source": f"《国家基本药物目录（2018年版）》第一部分 化学药品和生物制品，国家卫生健康委员会，PDF页码 P.{page_index}",
                }
            )
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in sorted(entries, key=lambda x: int(x["neml_no"])):
        if item["neml_no"] in seen:
            continue
        seen.add(item["neml_no"])
        deduped.append(item)
    return deduped


def build_template() -> list[dict[str, str]]:
    neml_entries = parse_neml_entries()
    rows: list[dict[str, str]] = []
    used_names: set[str] = set()
    for sequence, name in enumerate(P0_NAMES, 1):
        search_name = ALIASES.get(name, name)
        matched = next((item for item in neml_entries if search_name in item["name"] or item["name"] in search_name), None)
        rows.append(
            {
                "序号": str(sequence),
                "优先级": "P0",
                "类别": category_for(name),
                "药品通用名": name,
                "别名/商品名": "",
                "适应症": "",
                "规格": matched["spec"] if matched else "",
                "成人剂量": "",
                "特殊人群剂量": "",
                "用法": "",
                "禁忌": "",
                "不良反应": "",
                "注意事项": "",
                "来源": matched["source"] if matched else "P0清单；未在《国家基本药物目录（2018年版）》解析结果中定位，需说明书核对",
                "QA状态": "待说明书补全",
                "QA问题": "P0高风险条目，剂量需使用国家基本药物临床应用指南/CDE/NMPA/原厂说明书二核。",
                "NEML编号": matched["neml_no"] if matched else "",
                "NEML英文名": matched["english"] if matched else "",
            }
        )
        used_names.add(name)
    sequence = len(rows)
    for item in neml_entries:
        if sequence >= 400:
            break
        name = item["name"]
        if name in used_names or any(ALIASES.get(row["药品通用名"], row["药品通用名"]) == name for row in rows):
            continue
        sequence += 1
        rows.append(
            {
                "序号": str(sequence),
                "优先级": "P1",
                "类别": category_for(name),
                "药品通用名": name,
                "别名/商品名": "",
                "适应症": "",
                "规格": item["spec"],
                "成人剂量": "",
                "特殊人群剂量": "",
                "用法": "",
                "禁忌": "",
                "不良反应": "",
                "注意事项": "",
                "来源": item["source"],
                "QA状态": "待说明书补全",
                "QA问题": "P1扩展条目来自国家基本药物目录；具体适应症/剂量/禁忌需说明书二核。",
                "NEML编号": item["neml_no"],
                "NEML英文名": item["english"],
            }
        )
        used_names.add(name)
    return rows


def write_template(rows: list[dict[str, str]]) -> None:
    with TEMPLATE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fill_from_39ypk(rows: list[dict[str, str]], limit: int | None = None) -> list[dict[str, str]]:
    RAW_DIR.mkdir(exist_ok=True)
    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        if limit is not None and index > limit:
            output_rows.append(row)
            continue
        name = row["药品通用名"]
        try:
            candidate = ypk.choose_candidate(name)
            if not candidate:
                row["QA状态"] = "未匹配说明书"
                row["QA问题"] = row["QA问题"] + "；39药品通未匹配到候选，需CDE/NMPA/原厂说明书检索。"
                output_rows.append(row)
                continue
            manual, raw = ypk.extract_manual(candidate["url"])
            safe = re.sub(r'[\\/:*?"<>|]+', "_", f"{row['序号']}_{name}")
            (RAW_DIR / f"{safe}.html").write_text(raw, encoding="utf-8")
            original_source = row.get("来源", "")
            row["别名/商品名"] = manual["title"] or candidate["title"]
            row["适应症"] = manual["适应症"] or manual["功能主治"]
            row["规格"] = manual["规格"] or row["规格"]
            row["成人剂量"] = ypk.split_dose(manual["用法用量"])
            row["特殊人群剂量"] = ""
            row["用法"] = manual["用法用量"]
            row["禁忌"] = manual["禁忌"]
            row["不良反应"] = manual["不良反应"]
            row["注意事项"] = manual["注意事项"]
            ypk_source = "；".join(
                part
                for part in [
                    "39药品通说明书（非官方）",
                    manual["title"] or candidate["title"],
                    manual["manual_url"],
                    f"批准文号:{manual['批准文号']}" if manual["批准文号"] else "",
                    manual["生产企业"],
                ]
                if part
            )
            row["来源"] = original_source + "；" + ypk_source if original_source else ypk_source
            missing = [field for field in ["适应症", "规格", "成人剂量", "用法", "禁忌", "不良反应", "注意事项"] if not row.get(field)]
            if missing:
                row["QA状态"] = "字段不完整"
                row["QA问题"] = "缺字段:" + "、".join(missing) + "；非NMPA官网源，需二核；不可直接入训。"
            else:
                row["QA状态"] = "需NMPA/原厂二核"
                row["QA问题"] = "非NMPA官网源；剂量需与国家基本药物临床应用指南/CDE/NMPA/原厂说明书二核；不可直接入训。"
        except Exception as exc:
            row["QA状态"] = "抓取失败"
            row["QA问题"] = f"{type(exc).__name__}:{exc}；需CDE/NMPA/原厂说明书重新检索；不可直接入训。"
        output_rows.append(row)
        if index % 25 == 0:
            print("processed", index, "/", len(rows))
        time.sleep(0.2)
    return output_rows


def write_outputs(rows: list[dict[str, str]]) -> None:
    with DRAFT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
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
        "qa_status_counts": {},
        "official_base_source": str(NEML_PDF),
        "nonofficial_leaf_source": "39药品通说明书，仅作预填；需NMPA/CDE/原厂二核",
        "output_csv": str(OUTPUT),
        "qa_csv": str(QA_CSV),
        "raw_dir": str(RAW_DIR),
    }
    for row in rows:
        summary["qa_status_counts"][row["QA状态"]] = summary["qa_status_counts"].get(row["QA状态"], 0) + 1
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    rows = build_template()
    write_template(rows)
    filled_rows = fill_from_39ypk(rows)
    write_outputs(filled_rows)


if __name__ == "__main__":
    main()
