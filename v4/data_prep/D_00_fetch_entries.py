"""批量 fetch ydz entry 原文 (按 60 精确命中 entryId)
基于用户的 fetch_ydz.py entry() 函数, 加批量 + 进度
"""
import csv, json, re, html, time
import requests
from pathlib import Path

ROOT = Path("C:/Users/PC/medical_dose_research")
OUT = ROOT / "ydz_p0_entries_full.json"

BASE = "https://ydz.chp.org.cn/front-api/"
s = requests.Session()
s.headers.update({
    "Referer": "https://ydz.chp.org.cn/",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json;charset=utf-8",
})


def html_to_text(h):
    h = re.sub(r"</p\s*>", "\n", h or "", flags=re.I)
    h = re.sub(r"<br\s*/?>", "\n", h, flags=re.I)
    h = re.sub(r"<[^>]+>", "", h)
    return html.unescape(h).strip()


def fetch_entry(entry_id, retries=3):
    for attempt in range(retries):
        try:
            r = s.get(BASE + "entry/" + str(entry_id), timeout=30)
            r.raise_for_status()
            data = r.json()["data"]
            data["text"] = html_to_text(data.get("htmlContent"))
            return data
        except Exception as e:
            print(f"  ! entryId={entry_id} attempt {attempt+1}/{retries} fail: {e}", flush=True)
            time.sleep(1 + attempt)
    return None


# 读 P0 命中清单 - 提取 60 精确命中的 entryId
hits = []
with (ROOT / "ydz_p0_coverage.csv").open(encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["status"] != "官方药典精确命中":
            continue
        m = re.search(r"entryId=(\d+)", r["best_source"] or "")
        if not m:
            continue
        hits.append({
            "id": r["id"], "name": r["name"], "category": r["category"],
            "subcategory": r["subcategory"], "entryId": int(m.group(1)),
            "best_title": r["best_title"], "best_source": r["best_source"],
        })

print(f"[hits] {len(hits)} 精确命中, 唯一 entryId {len({h['entryId'] for h in hits})}")

# 已有的 4 条 selected 复用
existing = {}
sel_path = ROOT / "ydz_selected_entries.json"
if sel_path.exists():
    for e in json.loads(sel_path.read_text(encoding="utf-8")):
        existing[e["entryId"]] = e
    print(f"[reuse] {len(existing)} 条已 fetch")

# fetch 缺失
results = dict(existing)
todo = [h["entryId"] for h in hits if h["entryId"] not in existing]
print(f"[todo] {len(todo)} 条 entryId 待抓")

t0 = time.time()
for i, eid in enumerate(todo, 1):
    e = fetch_entry(eid)
    if e:
        results[eid] = e
    if i % 5 == 0 or i == len(todo):
        elapsed = time.time() - t0
        eta = elapsed / i * (len(todo) - i) if i > 0 else 0
        print(f"  [{i}/{len(todo)}] entryId={eid} title={e['title'] if e else 'FAIL'} elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
    time.sleep(0.3)

# 保存
out_list = list(results.values())
OUT.write_text(json.dumps(out_list, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[saved] {len(out_list)} entries → {OUT}")

# 统计原文长度
lengths = [len(e.get("text", "")) for e in out_list]
print(f"[text len] min={min(lengths)} max={max(lengths)} mean={sum(lengths)//len(lengths)}")
