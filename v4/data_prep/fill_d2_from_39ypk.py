import csv
import html
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests


INPUT = Path("D2_P0_80_western_meds_template.csv")
OUTPUT = Path("D2_P0_80_western_meds_39ypk_draft.csv")
RAW_DIR = Path("D2_39ypk_raw")

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
)


def clean_text(value: str) -> str:
    value = re.sub(r"<script[\s\S]*?</script>", "", value or "", flags=re.I)
    value = re.sub(r"<style[\s\S]*?</style>", "", value, flags=re.I)
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def request_text(url: str) -> str:
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def extract_candidates(text: str):
    pattern = re.compile(
        r'<a[^>]+href="(https://ypk\.39\.net/\d+/)"[^>]*>[\s\S]{0,900}?'
        r'class="commonly-drug-title">([\s\S]*?)</p>',
        re.I,
    )
    candidates = []
    for match in pattern.finditer(text):
        title = clean_text(match.group(2))
        if title:
            candidates.append({"title": title, "url": match.group(1)})
    if not candidates:
        # Search-result pages sometimes use different list markup.
        for href, title in re.findall(
            r'<a[^>]+href="(https://ypk\.39\.net/\d+/)"[^>]*>([\s\S]{0,120}?)</a>',
            text,
            re.I,
        ):
            title = clean_text(title)
            if title:
                candidates.append({"title": title, "url": href})
    seen = set()
    unique = []
    for item in candidates:
        if item["url"] not in seen:
            unique.append(item)
            seen.add(item["url"])
    return unique


def search_39ypk(keyword: str):
    return extract_candidates(request_text(f"https://ypk.39.net/search/{quote(keyword)}/"))


def normalized(text: str):
    return re.sub(r"[（）()ⅠⅡⅢIVⅣ\-\s/]+", "", text)


def core_name(name: str):
    name = re.sub(r"[（(].*?[）)]", "", name)
    for suffix in [
        "吸入气雾剂",
        "干混悬剂",
        "肠溶胶囊",
        "肠溶片",
        "控释片",
        "缓释片",
        "混悬液",
        "注射液",
        "贴剂",
        "胶囊",
        "颗粒",
        "软胶囊",
        "片",
        "钠",
        "钙",
        "酯",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)] or name
    return name


ALIASES = {
    "青霉素G": ["青霉素钠", "注射用青霉素钠"],
    "头孢哌酮舒巴坦": ["头孢哌酮钠舒巴坦钠", "注射用头孢哌酮钠舒巴坦钠"],
    "低分子肝素": ["低分子量肝素钙", "低分子肝素钙"],
    "口服补液盐III": ["口服补液盐散(Ⅲ)", "口服补液盐III"],
    "碳酸钙D3": ["碳酸钙D3片", "碳酸钙D3"],
    "布地奈德吸入剂": ["布地奈德吸入气雾剂", "布地奈德混悬液"],
    "异丙托溴铵": ["异丙托溴铵气雾剂", "吸入用异丙托溴铵溶液"],
    "噻托溴铵": ["噻托溴铵粉吸入剂", "噻托溴铵喷雾剂"],
    "枸橼酸铋钾": ["枸橼酸铋钾胶囊", "枸橼酸铋钾颗粒"],
    "芬太尼贴剂": ["芬太尼透皮贴剂"],
    "米索前列醇": ["米索前列醇片"],
    "门冬胰岛素": ["门冬胰岛素注射液"],
    "甘精胰岛素": ["甘精胰岛素注射液"],
    "阿卡波糖": ["阿卡波糖片"],
}


def score_candidate(title: str, query: str):
    score = 0
    nt = normalized(title)
    nq = normalized(query)
    core = normalized(core_name(query))
    if nq and nq in nt:
        score += 1000
    if core and core in nt:
        score += 500 + len(core)
    if nt and nq.startswith(nt):
        score += 100
    return score


def choose_candidate(name: str):
    queries = [name] + ALIASES.get(name, [])
    all_candidates = []
    for query in queries:
        try:
            all_candidates += [(query, item) for item in search_39ypk(query)]
        except Exception:
            pass
        time.sleep(0.1)
    scored = []
    for query, item in all_candidates:
        score = max(score_candidate(item["title"], q) for q in queries)
        if score:
            scored.append((score, query, item))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < 500:
        return None
    return scored[0][2]


def extract_field(text: str, field: str):
    match = re.search(
        rf'<p class="drug-explain-tit">【{re.escape(field)}】</p>\s*'
        rf'<p class="drug-explain-txt">([\s\S]*?)</p>',
        text,
        re.I,
    )
    if match:
        return clean_text(match.group(1))
    plain = clean_text(text)
    match = re.search(rf"【{re.escape(field)}】\s*(.*?)(?=【[^】]+】|$)", plain)
    return match.group(1).strip() if match else ""


def extract_manual(product_url: str):
    manual_url = product_url.rstrip("/") + "/manual/"
    text = request_text(manual_url)
    fields = {
        "manual_url": manual_url,
        "title": clean_text(re.search(r"<h1>([\s\S]*?)</h1>", text).group(1))
        if re.search(r"<h1>([\s\S]*?)</h1>", text)
        else "",
    }
    for key in ["适应症", "功能主治", "规格", "用法用量", "禁忌", "不良反应", "注意事项", "批准文号", "生产企业"]:
        fields[key] = extract_field(text, key)
    return fields, text


def split_dose(usage: str):
    return usage


def main():
    RAW_DIR.mkdir(exist_ok=True)
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8-sig", newline="")))
    output_rows = []
    for row in rows:
        name = row["药品通用名"]
        try:
            candidate = choose_candidate(name)
            if not candidate:
                row["QA状态"] = "未匹配说明书"
                row["QA问题"] = "39药品通未匹配到可信候选；需CDE/NMPA/原厂说明书检索"
                output_rows.append(row)
                continue
            manual, raw = extract_manual(candidate["url"])
            safe = re.sub(r'[\\/:*?"<>|]+', "_", f"{row['序号']}_{name}")
            (RAW_DIR / f"{safe}.html").write_text(raw, encoding="utf-8")
            row["别名/商品名"] = manual["title"] or candidate["title"]
            row["适应症"] = manual["适应症"] or manual["功能主治"]
            row["规格"] = manual["规格"]
            row["成人剂量"] = split_dose(manual["用法用量"])
            row["特殊人群剂量"] = ""
            row["用法"] = manual["用法用量"]
            row["禁忌"] = manual["禁忌"]
            row["不良反应"] = manual["不良反应"]
            row["注意事项"] = manual["注意事项"]
            row["来源"] = "；".join(
                part
                for part in [
                    "39药品通说明书",
                    manual["title"] or candidate["title"],
                    manual["manual_url"],
                    f"批准文号:{manual['批准文号']}" if manual["批准文号"] else "",
                    manual["生产企业"],
                ]
                if part
            )
            missing = [field for field in ["适应症", "规格", "成人剂量", "禁忌", "不良反应", "注意事项"] if not row.get(field)]
            if missing:
                row["QA状态"] = "字段不完整"
                row["QA问题"] = "缺字段:" + "、".join(missing) + "；非NMPA官网源，需二核"
            else:
                row["QA状态"] = "需NMPA/原厂二核"
                row["QA问题"] = "非NMPA官网源；剂量需与国家基本药物临床应用指南/CDE/NMPA/原厂说明书二核"
            output_rows.append(row)
        except Exception as exc:
            row["QA状态"] = "抓取失败"
            row["QA问题"] = f"{type(exc).__name__}:{exc}"
            output_rows.append(row)
        time.sleep(0.25)

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print("total", len(output_rows))
    for status in sorted(set(row["QA状态"] for row in output_rows)):
        print(status, sum(row["QA状态"] == status for row in output_rows))
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
