from __future__ import annotations

import csv
import json
import re
from pathlib import Path


BOOK_SOURCE = "《方剂学》第十版（邓中甲主编，人民卫生出版社，2016，ISBN 9787117218900）"
PRINT_TO_PDF_OFFSET = 51


def find_base_dir() -> Path:
    home = Path.home()
    for child in home.iterdir():
        if child.is_dir() and "Claude" in child.name:
            candidate = child / "medical_sft" / "v4" / "data_prep"
            if candidate.exists():
                return candidate
    raise FileNotFoundError("未找到 medical_sft/v4/data_prep 目录")


BASE = find_base_dir()
TEXT_DIR = BASE / "fangjixue10_pages_text"
INPUT_CSV = BASE / "D1_P0_119_formula_textbook_template_with_ydz_candidates.csv"
OUT_CSV = BASE / "D1_P0_119_formula_textbook_extracted.csv"
QA_CSV = BASE / "D1_P0_119_formula_textbook_QA.csv"
SNIPPET_DIR = BASE / "D1_formula_snippets"


SECTION_TAGS = [
    "【组成】",
    "【用法】",
    "【功用】",
    "【主治】",
    "【证治机理】",
    "【方解】",
    "【配伍特点】",
    "【运用】",
    "【附方】",
    "【鉴别】",
    "【方论选录】",
    "【医案举例】",
    "【方歌】",
    "【复习思考题】",
]

CAUTION_KEYWORDS = [
    "禁",
    "忌",
    "慎",
    "不可",
    "不宜",
    "勿",
    "孕妇",
    "年老",
    "体弱",
    "服后",
    "过剂",
    "中病即止",
    "不当",
]


def read_pages() -> dict[int, str]:
    pages: dict[int, str] = {}
    for path in sorted(TEXT_DIR.glob("page_*.txt")):
        match = re.search(r"(\d+)", path.stem)
        if not match:
            continue
        pages[int(match.group(1))] = path.read_text("utf-8", errors="replace")
    return pages


def normalize_spaces(text: str) -> str:
    text = (text or "").replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def compact_line(text: str) -> str:
    return re.sub(r"[\s\u3000]+", "", text or "")


def clean_field(text: str, max_chars: int | None = None) -> str:
    text = normalize_spaces(text)
    text = text.replace("\n", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ：:；;，,")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip(" ，,；;。") + "……"
    return text


def strip_tag(text: str, tag: str) -> str:
    return text.replace(tag, "", 1).strip()


def extract_between(block: str, start_tag: str, end_tags: list[str] | None = None) -> str:
    start = block.find(start_tag)
    if start < 0:
        return ""
    start += len(start_tag)
    if end_tags is None:
        end_tags = [tag for tag in SECTION_TAGS if tag != start_tag]
    end_positions = [block.find(tag, start) for tag in end_tags]
    end_positions = [pos for pos in end_positions if pos >= 0]
    end = min(end_positions) if end_positions else len(block)
    return clean_field(block[start:end])


def split_sentences(text: str) -> list[str]:
    text = clean_field(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[。！？；;])", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def extract_cautions(block: str) -> str:
    candidates: list[str] = []
    for tag in ["【用法】", "【运用】", "【主治】"]:
        section = extract_between(block, tag)
        for sentence in split_sentences(section):
            if any(keyword in sentence for keyword in CAUTION_KEYWORDS):
                candidates.append(sentence)
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in candidates:
        if sentence not in seen:
            seen.add(sentence)
            deduped.append(sentence)
    return clean_field(" ".join(deduped), 500)


def build_index_text(pages: dict[int, str]) -> str:
    return "\n".join(pages.get(page, "") for page in range(536, 554))


def index_pages_for_name(index_text: str, name: str) -> list[int]:
    pages: list[int] = []
    for raw_line in index_text.splitlines():
        line = raw_line.strip()
        if not line or line.endswith("画") or "附录" in line:
            continue
        match = re.match(r"(.+?)[　 ]+(\d{1,3})$", line)
        if not match:
            continue
        entry_name = match.group(1).strip()
        page = int(match.group(2))
        if entry_name == name or entry_name.startswith(f"{name}（") or entry_name.startswith(f"{name}("):
            if page not in pages:
                pages.append(page)
    return pages


def title_matches(line: str, name: str) -> bool:
    line_norm = compact_line(line)
    name_norm = compact_line(name)
    if line_norm == name_norm:
        return True
    if line_norm.startswith(name_norm + "（"):
        return True
    if line_norm.startswith(name_norm + "("):
        return True
    return False


def source_after_title(lines: list[str], line_index: int) -> tuple[str, bool]:
    window: list[str] = []
    has_composition = False
    for next_line in lines[line_index + 1 : line_index + 16]:
        stripped = next_line.strip()
        if not stripped:
            continue
        if "【组成】" in stripped:
            has_composition = True
            break
        window.append(stripped)
    source = "".join(window)
    source = clean_field(source, 120)
    return source, has_composition


def find_title_on_page(page_text: str, name: str) -> tuple[int, str, str] | None:
    lines = page_text.splitlines()
    for line_index, line in enumerate(lines):
        if not title_matches(line, name):
            continue
        source, has_composition = source_after_title(lines, line_index)
        if has_composition and "《" in source and "》" in source:
            char_pos = sum(len(item) + 1 for item in lines[:line_index])
            return char_pos, line.strip(), source
    return None


def find_title_in_pages(
    pages: dict[int, str], start_page: int, name: str, window_pages: int = 3
) -> dict[str, object] | None:
    lines: list[tuple[int, int, int, str]] = []
    for page in range(start_page, start_page + window_pages):
        page_text = pages.get(page, "")
        offset = 0
        for line_number, line in enumerate(page_text.splitlines(), 1):
            lines.append((page, line_number, offset, line))
            offset += len(line) + 1
    text_lines = [line for _, _, _, line in lines]
    for line_index, (page, _line_number, char_pos, line) in enumerate(lines):
        if not title_matches(line, name):
            continue
        source, has_composition = source_after_title(text_lines, line_index)
        if has_composition and "《" in source and "》" in source:
            return {
                "pdf_page": page,
                "char_pos": char_pos,
                "located_title": line.strip(),
                "title_source": source,
                "method": "索引页码定位",
            }
    return None


def inline_affix_match(text: str, name: str) -> re.Match[str] | None:
    patterns = [
        rf"(?:\d+[.．、]\s*)?{re.escape(name)}\s*[（(]([^）)]*《[^）)]*》[^）)]*)[）)]\s*[　 ]*",
        rf"(?:【附方】\s*)?{re.escape(name)}\s*[（(]([^）)]*《[^）)]*》[^）)]*)[）)]\s*[　 ]*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match
    return None


def extract_inline_affix(text: str, name: str) -> dict[str, str] | None:
    match = inline_affix_match(text, name)
    if not match:
        return None
    start = match.start()
    entry = text[match.end() :]
    next_match = re.search(r"\n?\s*\d+[.．、]\s*[\u4e00-\u9fa5]{2,12}\s*[（(]《", entry)
    end = next_match.start() if next_match else len(entry)
    entry = entry[:end]
    entry = re.split(r"【鉴别】|【方论选录】|【医案举例】|【方歌】", entry, maxsplit=1)[0]
    entry_clean = clean_field(entry)
    if not entry_clean:
        return None
    source = clean_field(f"《{match.group(1).split('《', 1)[1].split('》', 1)[0]}》")
    function = ""
    indications = ""
    usage = ""
    composition = entry_clean
    function_match = re.search(r"功用[:：]\s*(.+?)(?=主治[:：]|$)", composition)
    if function_match:
        function = clean_field(function_match.group(1))
        composition = clean_field(composition[: function_match.start()])
    indications_match = re.search(r"主治[:：]\s*(.+)$", entry_clean)
    if indications_match:
        indications = clean_field(indications_match.group(1))
    usage_match = re.search(r"(?:上|以水|水[^药味]*煮|水煎服|为丸|每服|先煮)", composition)
    if usage_match:
        usage = clean_field(composition[usage_match.start() :])
        composition = clean_field(composition[: usage_match.start()])
    return {
        "source": source,
        "composition": composition,
        "usage": usage,
        "function": function,
        "indications": indications,
        "raw": clean_field(entry, 1200),
        "start": str(start),
    }


def find_inline_affix(
    pages: dict[int, str], name: str, print_pages: list[int]
) -> dict[str, object] | None:
    candidate_pdf_pages: list[int] = []
    if print_pages:
        for print_page in print_pages:
            for pdf_page in range(35, 536):
                if abs(pdf_page - (print_page + PRINT_TO_PDF_OFFSET)) <= 35:
                    candidate_pdf_pages.append(pdf_page)
    candidate_pdf_pages.extend(range(35, 536))
    seen: set[int] = set()
    for pdf_page in candidate_pdf_pages:
        if pdf_page in seen:
            continue
        seen.add(pdf_page)
        text = "\n".join(pages.get(page, "") for page in range(pdf_page, pdf_page + 3))
        parsed = extract_inline_affix(text, name)
        if parsed:
            return {
                "pdf_page": pdf_page,
                "char_pos": 0,
                "located_title": name,
                "title_source": parsed["source"],
                "method": "附方段落定位",
                "inline": parsed,
            }
    return None


def find_formula_start(
    pages: dict[int, str], name: str, print_pages: list[int]
) -> dict[str, object] | None:
    candidate_pdf_pages: list[int] = []
    for print_page in print_pages:
        for pdf_page in range(print_page + PRINT_TO_PDF_OFFSET - 2, print_page + PRINT_TO_PDF_OFFSET + 3):
            if pdf_page in pages and pdf_page not in candidate_pdf_pages:
                candidate_pdf_pages.append(pdf_page)
    if not candidate_pdf_pages:
        candidate_pdf_pages = [page for page in sorted(pages) if 35 <= page <= 535]

    for pdf_page in candidate_pdf_pages:
        found = find_title_in_pages(pages, pdf_page, name)
        if found:
            found["method"] = "索引页码定位" if print_pages else "全文标题定位"
            return found

    for pdf_page in sorted(pages):
        if pdf_page < 35 or pdf_page > 535:
            continue
        found = find_title_in_pages(pages, pdf_page, name)
        if found:
            found["method"] = "全文标题定位"
            return found

    return find_inline_affix(pages, name, print_pages)


def block_from_start(pages: dict[int, str], pdf_page: int, char_pos: int, max_pages: int = 8) -> str:
    pieces: list[str] = []
    for page in range(pdf_page, pdf_page + max_pages):
        text = pages.get(page, "")
        if not text:
            continue
        if page == pdf_page:
            text = text[char_pos:]
        pieces.append(text)
    return "\n".join(pieces)


def main() -> None:
    pages = read_pages()
    index_text = build_index_text(pages)
    SNIPPET_DIR.mkdir(exist_ok=True)

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        input_fields = reader.fieldnames or []
        rows = list(reader)

    extra_fields = [
        "定位标题",
        "PDF页码",
        "书内索引页码",
        "页码类型",
        "定位方法",
        "原文片段文件",
    ]
    out_fields = input_fields + [field for field in extra_fields if field not in input_fields]

    qa_fields = [
        "序号",
        "章节",
        "方剂名",
        "QA状态",
        "QA问题",
        "PDF页码",
        "书内索引页码",
        "缺失字段",
        "定位标题",
        "定位方法",
        "原文片段文件",
    ]

    output_rows: list[dict[str, str]] = []
    qa_rows: list[dict[str, str]] = []

    for row in rows:
        name = row["方剂名"]
        print_pages = index_pages_for_name(index_text, name)
        location = find_formula_start(pages, name, print_pages)

        out_row = dict(row)
        qa_problem: list[str] = []

        if not location:
            out_row.update(
                {
                    "出处": "",
                    "组成": "",
                    "用法": "",
                    "功用": "",
                    "主治": "",
                    "方义": "",
                    "配伍特点": "",
                    "注意/禁忌": "",
                    "《方剂学》页码": "",
                    "来源": BOOK_SOURCE,
                    "QA状态": "未定位",
                    "QA问题": "未在教材PDF正文中定位到“方名→出处→【组成】”结构；需人工核对是否为本版附方/异名/未收方。",
                    "定位标题": "",
                    "PDF页码": "",
                    "书内索引页码": "；".join(f"P.{page}" for page in print_pages),
                    "页码类型": "未定位",
                    "定位方法": "",
                    "原文片段文件": "",
                }
            )
            output_rows.append(out_row)
            qa_rows.append(
                {
                    "序号": row["序号"],
                    "章节": row["章节"],
                    "方剂名": name,
                    "QA状态": "未定位",
                    "QA问题": out_row["QA问题"],
                    "PDF页码": "",
                    "书内索引页码": out_row["书内索引页码"],
                    "缺失字段": "出处；组成；用法；功用；主治",
                    "定位标题": "",
                    "定位方法": "",
                    "原文片段文件": "",
                }
            )
            continue

        pdf_page = int(location["pdf_page"])
        char_pos = int(location["char_pos"])
        block = block_from_start(pages, pdf_page, char_pos)
        snippet_path = SNIPPET_DIR / f"{int(row['序号']):03d}_{name}.txt"
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
            composition = extract_between(block, "【组成】", ["【用法】"])
            usage = extract_between(block, "【用法】", ["【功用】"])
            function = extract_between(block, "【功用】", ["【主治】"])
            indications = extract_between(block, "【主治】", ["【证治机理】", "【方解】"])
            formula_explain = extract_between(block, "【方解】", ["【配伍特点】", "【运用】", "【附方】", "【鉴别】", "【方论选录】", "【方歌】"])
            formula_explain = clean_field(formula_explain, 220)
            compatibility = extract_between(block, "【配伍特点】", ["【运用】", "【附方】", "【鉴别】", "【方论选录】", "【方歌】"])
            compatibility = clean_field(compatibility, 220)
            cautions = extract_cautions(block)

        print_page_text = "；".join(f"P.{page}" for page in print_pages)
        out_row.update(
            {
                "出处": title_source,
                "组成": composition,
                "用法": usage,
                "功用": function,
                "主治": indications,
                "方义": formula_explain,
                "配伍特点": compatibility,
                "注意/禁忌": cautions,
                "《方剂学》页码": print_page_text or f"PDF P.{pdf_page}",
                "来源": f"{BOOK_SOURCE}；PDF页码 P.{pdf_page}"
                + (f"；书内索引页码 {print_page_text}" if print_page_text else ""),
                "定位标题": str(location["located_title"]),
                "PDF页码": f"P.{pdf_page}",
                "书内索引页码": print_page_text,
                "页码类型": "PDF页码+书内索引页码" if print_page_text else "PDF页码",
                "定位方法": str(location["method"]),
                "原文片段文件": str(snippet_path.relative_to(BASE)),
            }
        )

        required = {
            "出处": title_source,
            "组成": composition,
            "用法": usage,
            "功用": function,
            "主治": indications,
        }
        missing = [field for field, value in required.items() if not value]
        if missing:
            qa_status = "字段缺失"
            qa_problem.append("缺失强制字段：" + "、".join(missing))
        else:
            qa_status = "已从教材PDF抽取待人工复核"

        if len(print_pages) > 1:
            qa_problem.append("方名索引存在多个页码/同名方，已抽取首个正文定位，需人工确认版本。")
            qa_status = "同名/多出处需复核" if qa_status != "字段缺失" else qa_status

        if not cautions:
            qa_problem.append("教材正文未抽到明确注意/禁忌句，需人工补核。")
            if qa_status == "已从教材PDF抽取待人工复核":
                qa_status = "注意/禁忌待补核"

        out_row["QA状态"] = qa_status
        out_row["QA问题"] = "；".join(qa_problem) if qa_problem else "字段来自教材PDF文本抽取，训练前仍需人工核对剂量与页码。"

        output_rows.append(out_row)
        qa_rows.append(
            {
                "序号": row["序号"],
                "章节": row["章节"],
                "方剂名": name,
                "QA状态": qa_status,
                "QA问题": out_row["QA问题"],
                "PDF页码": out_row["PDF页码"],
                "书内索引页码": out_row["书内索引页码"],
                "缺失字段": "；".join(missing),
                "定位标题": out_row["定位标题"],
                "定位方法": out_row["定位方法"],
                "原文片段文件": out_row["原文片段文件"],
            }
        )

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    with QA_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=qa_fields)
        writer.writeheader()
        writer.writerows(qa_rows)

    summary = {
        "total": len(output_rows),
        "qa_status_counts": {},
        "output_csv": str(OUT_CSV),
        "qa_csv": str(QA_CSV),
        "snippet_dir": str(SNIPPET_DIR),
    }
    for row in output_rows:
        summary["qa_status_counts"][row["QA状态"]] = summary["qa_status_counts"].get(row["QA状态"], 0) + 1
    summary_path = BASE / "D1_P0_119_formula_textbook_extracted_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
