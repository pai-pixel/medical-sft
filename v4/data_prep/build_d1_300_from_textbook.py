from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import extract_d1_textbook as d1


OUT_CSV = d1.BASE / "D1_300_formula_textbook_extracted.csv"
QA_CSV = d1.BASE / "D1_300_formula_textbook_QA.csv"
SUMMARY_JSON = d1.BASE / "D1_300_formula_textbook_summary.json"
SNIPPET_DIR = d1.BASE / "D1_formula_snippets_300"


CHAPTERS = [
    (72, "第1章 解表剂"),
    (110, "第2章 泻下剂"),
    (135, "第3章 和解剂"),
    (151, "第4章 清热剂"),
    (191, "第5章 祛暑剂"),
    (201, "第6章 温里剂"),
    (221, "第7章 表里双解剂"),
    (231, "第8章 补益剂"),
    (279, "第9章 固涩剂"),
    (300, "第10章 安神剂"),
    (317, "第11章 开窍剂"),
    (330, "第12章 理气剂"),
    (357, "第13章 理血剂"),
    (385, "第14章 治风剂"),
    (407, "第15章 治燥剂"),
    (424, "第16章 祛湿剂"),
    (468, "第17章 祛痰剂"),
    (491, "第18章 消食剂"),
    (500, "第19章 驱虫剂"),
    (507, "第20章 涌吐剂"),
    (514, "第21章 治痈疡剂"),
]


def chapter_for_pdf_page(pdf_page: int) -> str:
    current = CHAPTERS[0][1]
    for start_page, chapter in CHAPTERS:
        if pdf_page >= start_page:
            current = chapter
        else:
            break
    return current


def base_name(index_name: str) -> str:
    return re.sub(r"[（(].*?[）)]", "", index_name).strip()


def index_entries(index_text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for raw_line in index_text.splitlines():
        line = raw_line.strip()
        if not line or line.endswith("画") or "附录" in line:
            continue
        match = re.match(r"(.+?)[　 ]+(\d{1,3})$", line)
        if not match:
            continue
        indexed_name = match.group(1).strip()
        print_page = int(match.group(2))
        key = (indexed_name, print_page)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "indexed_name": indexed_name,
                "name": base_name(indexed_name),
                "print_page": print_page,
            }
        )
    return rows


def next_index_page(entries: list[dict[str, object]], entry_index: int) -> int | None:
    page = int(entries[entry_index]["print_page"])
    later_pages = [int(item["print_page"]) for item in entries[entry_index + 1 :] if int(item["print_page"]) > page]
    return min(later_pages) if later_pages else None


def infer_candidate_pdf_pages(print_page: int) -> list[int]:
    first = print_page + d1.PRINT_TO_PDF_OFFSET
    pages = list(range(first - 2, first + 4))
    return [page for page in pages if 35 <= page <= 535]


def locate_formula(pages: dict[int, str], name: str, print_page: int) -> dict[str, object] | None:
    location = d1.find_formula_start(pages, name, [print_page])
    if location:
        return location
    # Some index entries include a distinguishing bracketed source. Try exact indexed title as a fallback.
    return None


def parse_row(
    pages: dict[int, str],
    sequence: int,
    p_level: str,
    chapter: str,
    name: str,
    location: dict[str, object] | None,
    print_page: int,
) -> dict[str, str]:
    row = {
        "序号": str(sequence),
        "优先级": p_level,
        "章节": chapter,
        "方剂名": name,
        "出处": "",
        "组成": "",
        "用法": "",
        "功用": "",
        "主治": "",
        "方义": "",
        "配伍特点": "",
        "注意/禁忌": "",
        "《方剂学》页码": f"P.{print_page}" if print_page else "",
        "来源": d1.BOOK_SOURCE,
        "QA状态": "未定位",
        "QA问题": "未在教材PDF正文中定位到“方名→出处→【组成】”结构；需人工核对是否为本版附方/异名/未收方。",
        "定位标题": "",
        "PDF页码": "",
        "书内索引页码": f"P.{print_page}" if print_page else "",
        "页码类型": "未定位",
        "定位方法": "",
        "原文片段文件": "",
    }
    if not location:
        return row

    pdf_page = int(location["pdf_page"])
    char_pos = int(location["char_pos"])
    block = d1.block_from_start(pages, pdf_page, char_pos)
    snippet_path = SNIPPET_DIR / f"{sequence:03d}_{name}.txt"
    snippet_path.write_text(block[:6000], "utf-8")

    title_source = str(location["title_source"])
    inline = location.get("inline")
    if isinstance(inline, dict):
        composition = str(inline.get("composition", ""))
        usage = str(inline.get("usage", ""))
        function = str(inline.get("function", ""))
        indications = str(inline.get("indications", ""))
        formula_explain = ""
        compatibility = ""
        cautions = ""
    else:
        composition = d1.extract_between(block, "【组成】", ["【用法】"])
        usage = d1.extract_between(block, "【用法】", ["【功用】"])
        function = d1.extract_between(block, "【功用】", ["【主治】"])
        indications = d1.extract_between(block, "【主治】", ["【证治机理】", "【方解】"])
        formula_explain = d1.clean_field(
            d1.extract_between(block, "【方解】", ["【配伍特点】", "【运用】", "【附方】", "【鉴别】", "【方论选录】", "【方歌】"]),
            220,
        )
        compatibility = d1.clean_field(
            d1.extract_between(block, "【配伍特点】", ["【运用】", "【附方】", "【鉴别】", "【方论选录】", "【方歌】"]),
            220,
        )
        cautions = d1.extract_cautions(block)

    row.update(
        {
            "出处": title_source,
            "组成": composition,
            "用法": usage,
            "功用": function,
            "主治": indications,
            "方义": formula_explain,
            "配伍特点": compatibility,
            "注意/禁忌": cautions,
            "《方剂学》页码": f"P.{print_page}" if print_page else f"PDF P.{pdf_page}",
            "来源": f"{d1.BOOK_SOURCE}；PDF页码 P.{pdf_page}" + (f"；书内索引页码 P.{print_page}" if print_page else ""),
            "定位标题": str(location["located_title"]),
            "PDF页码": f"P.{pdf_page}",
            "书内索引页码": f"P.{print_page}" if print_page else "",
            "页码类型": "PDF页码+书内索引页码" if print_page else "PDF页码",
            "定位方法": str(location["method"]),
            "原文片段文件": str(snippet_path.relative_to(d1.BASE)),
        }
    )

    required = {"出处": title_source, "组成": composition, "用法": usage, "功用": function, "主治": indications}
    missing = [field for field, value in required.items() if not value]
    qa_problem: list[str] = []
    if missing:
        row["QA状态"] = "字段缺失"
        qa_problem.append("缺失强制字段：" + "、".join(missing))
    else:
        row["QA状态"] = "已从教材PDF抽取待人工复核"
    if not cautions:
        qa_problem.append("教材正文未抽到明确注意/禁忌句，需人工补核。")
        if row["QA状态"] == "已从教材PDF抽取待人工复核":
            row["QA状态"] = "注意/禁忌待补核"
    if p_level == "P1":
        qa_problem.append("P1扩展条目由书后方名索引自动抽取，训练前需人工逐条核对。")
    row["QA问题"] = "；".join(qa_problem) if qa_problem else "字段来自教材PDF文本抽取，训练前仍需人工核对剂量与页码。"
    return row


def main() -> None:
    pages = d1.read_pages()
    index_text = d1.build_index_text(pages)
    entries = index_entries(index_text)
    SNIPPET_DIR.mkdir(exist_ok=True)

    p0_rows = list(csv.DictReader(d1.OUT_CSV.open("r", encoding="utf-8-sig", newline="")))
    p0_names = {row["方剂名"] for row in p0_rows}
    output_rows: list[dict[str, str]] = []
    for sequence, row in enumerate(p0_rows, 1):
        out = {
            "序号": str(sequence),
            "优先级": "P0",
            "章节": row["章节"],
            "方剂名": row["方剂名"],
            "出处": row["出处"],
            "组成": row["组成"],
            "用法": row["用法"],
            "功用": row["功用"],
            "主治": row["主治"],
            "方义": row["方义"],
            "配伍特点": row["配伍特点"],
            "注意/禁忌": row["注意/禁忌"],
            "《方剂学》页码": row["《方剂学》页码"],
            "来源": row["来源"],
            "QA状态": row["QA状态"],
            "QA问题": row["QA问题"],
            "定位标题": row["定位标题"],
            "PDF页码": row["PDF页码"],
            "书内索引页码": row["书内索引页码"],
            "页码类型": row["页码类型"],
            "定位方法": row["定位方法"],
            "原文片段文件": row["原文片段文件"],
        }
        output_rows.append(out)

    sequence = len(output_rows)
    skipped_missing: list[str] = []
    for entry in entries:
        if sequence >= 300:
            break
        name = str(entry["name"])
        if name in p0_names or any(row["方剂名"] == name for row in output_rows):
            continue
        print_page = int(entry["print_page"])
        location = locate_formula(pages, name, print_page)
        if not location:
            continue
        chapter = chapter_for_pdf_page(int(location["pdf_page"]))
        candidate = parse_row(pages, sequence + 1, "P1", chapter, name, location, print_page)
        if candidate["QA状态"] == "字段缺失":
            skipped_missing.append(name)
            continue
        sequence += 1
        output_rows.append(candidate)

    fieldnames = [
        "序号",
        "优先级",
        "章节",
        "方剂名",
        "出处",
        "组成",
        "用法",
        "功用",
        "主治",
        "方义",
        "配伍特点",
        "注意/禁忌",
        "《方剂学》页码",
        "来源",
        "QA状态",
        "QA问题",
        "定位标题",
        "PDF页码",
        "书内索引页码",
        "页码类型",
        "定位方法",
        "原文片段文件",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    qa_fields = ["序号", "优先级", "章节", "方剂名", "QA状态", "QA问题", "PDF页码", "书内索引页码", "定位方法", "原文片段文件"]
    with QA_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=qa_fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in qa_fields})

    summary = {
        "total": len(output_rows),
        "p0": sum(1 for row in output_rows if row["优先级"] == "P0"),
        "p1": sum(1 for row in output_rows if row["优先级"] == "P1"),
        "qa_status_counts": {},
        "indexed_formula_entries": len(entries),
        "skipped_p1_missing_required_fields": len(skipped_missing),
        "skipped_p1_missing_required_field_names": skipped_missing[:80],
        "output_csv": str(OUT_CSV),
        "qa_csv": str(QA_CSV),
        "snippet_dir": str(SNIPPET_DIR),
    }
    for row in output_rows:
        summary["qa_status_counts"][row["QA状态"]] = summary["qa_status_counts"].get(row["QA状态"], 0) + 1
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
