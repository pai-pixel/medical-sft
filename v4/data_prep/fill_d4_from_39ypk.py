import csv
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests


INPUT = Path("D4_60_supplement_template.csv")
OUTPUT = Path("D4_60_supplement_filled_39ypk.csv")
RAW_DIR = Path("D4_39ypk_raw")

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
)


def clean_text(value: str) -> str:
    value = re.sub(r"<script[\s\S]*?</script>", "", value, flags=re.I)
    value = re.sub(r"<style[\s\S]*?</style>", "", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def request_text(url: str) -> str:
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def search_39ypk(keyword: str):
    url = f"https://ypk.39.net/search/{quote(keyword)}/"
    text = request_text(url)
    links = []
    for href, inner in re.findall(r'<a[^>]+href="(https://ypk\.39\.net/\d+/)"[^>]*>([\s\S]*?)</a>', text, re.I):
        title_match = re.search(r'class="commonly-drug-title">([\s\S]*?)</p>', inner, re.I)
        title = clean_text(title_match.group(1) if title_match else inner)
        if title:
            links.append({"title": title, "url": href})
    seen = set()
    unique_links = []
    for link in links:
        if link["url"] not in seen:
            unique_links.append(link)
            seen.add(link["url"])
    return unique_links


def extract_field(text: str, field: str) -> str:
    pattern = (
        rf'<p class="drug-explain-tit">【{re.escape(field)}】</p>\s*'
        rf'<p class="drug-explain-txt">([\s\S]*?)</p>'
    )
    match = re.search(pattern, text, re.I)
    if match:
        return clean_text(match.group(1))
    # fallback for pages whose tags differ
    text_plain = clean_text(text)
    match = re.search(rf"【{re.escape(field)}】\s*(.*?)(?=【[^】]+】|$)", text_plain)
    return match.group(1).strip() if match else ""


def extract_manual(product_url: str):
    manual_url = product_url.rstrip("/") + "/manual/"
    text = request_text(manual_url)
    result = {
        "manual_url": manual_url,
        "不良反应": extract_field(text, "不良反应"),
        "禁忌": extract_field(text, "禁忌"),
        "注意事项": extract_field(text, "注意事项"),
        "批准文号": extract_field(text, "批准文号"),
        "生产企业": extract_field(text, "生产企业"),
    }
    return result, text


def core_tokens(text: str):
    text = re.sub(r"[（(].*?[）)]", "", text)
    suffixes = [
        "吸入气雾剂",
        "干混悬剂",
        "肠溶胶囊",
        "肠溶片",
        "缓释片",
        "注射液",
        "软胶囊",
        "胶囊",
        "颗粒",
        "散",
        "丸",
        "片",
        "钠",
        "钙",
    ]
    tokens = [text]
    for suffix in suffixes:
        if text.endswith(suffix):
            tokens.append(text[: -len(suffix)])
            tokens.append(suffix)
            break
    return [token for token in tokens if token]


def score_candidate(title: str, keyword: str, fallback_keyword: str = ""):
    score = 0
    if title == keyword:
        score += 1000
    if keyword and keyword in title:
        score += 500
    if title and keyword.startswith(title):
        score += 250
    for token in core_tokens(keyword):
        if token and token in title:
            score += 100 + len(token)
    for token in core_tokens(fallback_keyword):
        if token and token in title:
            score += 50 + len(token)
    return score


def choose_candidate(keyword: str, candidates, fallback_keyword: str = ""):
    if not candidates:
        return None
    scored = [
        (score_candidate(item["title"], keyword, fallback_keyword), item)
        for item in candidates
        if item.get("title")
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if not scored or scored[0][0] < 100:
        return None
    return scored[0][1]


def main():
    RAW_DIR.mkdir(exist_ok=True)
    rows = []
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            keyword = row["药典通用名"].strip() or row["P0 检索名"].strip()
            fallback_keyword = row["P0 检索名"].strip()
            try:
                candidates = search_39ypk(keyword)
                candidate = choose_candidate(keyword, candidates, fallback_keyword)
                if not candidate and fallback_keyword != keyword:
                    candidates = search_39ypk(fallback_keyword)
                    candidate = choose_candidate(fallback_keyword, candidates, keyword)
                if not candidate:
                    row["备注"] = (row.get("备注", "") + "；39药品通未检索到候选").strip("；")
                    rows.append(row)
                    continue
                manual, raw_text = extract_manual(candidate["url"])
                safe_name = re.sub(r'[\\/:*?"<>|]+', "_", f"{row['序号']}_{keyword}")
                (RAW_DIR / f"{safe_name}.html").write_text(raw_text, encoding="utf-8")
                row["待补_禁忌"] = manual["禁忌"]
                row["待补_不良反应"] = manual["不良反应"]
                row["待补_注意事项"] = manual["注意事项"]
                source_parts = ["39药品通说明书", candidate["title"], manual["manual_url"]]
                if manual["批准文号"]:
                    source_parts.append(f"批准文号:{manual['批准文号']}")
                if manual["生产企业"]:
                    source_parts.append(manual["生产企业"])
                row["说明书来源"] = "；".join(source_parts)
                note = "非NMPA官网源，需国家药监局/原厂说明书二次核验"
                row["备注"] = (row.get("备注", "") + "；" + note).strip("；")
                rows.append(row)
            except Exception as exc:
                row["备注"] = (row.get("备注", "") + f"；抓取失败:{type(exc).__name__}:{exc}").strip("；")
                rows.append(row)
            time.sleep(0.4)

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total": len(rows),
        "filled_禁忌": sum(bool(row.get("待补_禁忌")) for row in rows),
        "filled_不良反应": sum(bool(row.get("待补_不良反应")) for row in rows),
        "filled_注意事项": sum(bool(row.get("待补_注意事项")) for row in rows),
        "output": str(OUTPUT),
    }
    Path("D4_60_supplement_filled_39ypk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
