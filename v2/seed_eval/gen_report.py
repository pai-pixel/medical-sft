"""生成 HTML 报告: 35 题对照,题目+参考答案+模型答案+key_points 命中率,
便于浏览器打开滑动审阅。"""
import json
import re
import html

ROWS = []
with open(r"C:\Users\PC\Claude脚本\medical_v2_eval\seed_inference_out.jsonl", encoding="utf-8") as f:
    for line in f:
        ROWS.append(json.loads(line))

def hit_rate(row):
    ans = row["model_answer"]
    kps = row["key_points"]
    if not kps:
        return None, [], []
    hits, miss = [], []
    for kp in kps:
        alts = re.split(r"[（）()]", kp)
        alts = [a.strip() for a in alts if a.strip()]
        if any(a in ans for a in alts):
            hits.append(kp)
        else:
            miss.append(kp)
    return len(hits) / len(kps), hits, miss

# 整体统计
rates = [hit_rate(r)[0] for r in ROWS if hit_rate(r)[0] is not None]
avg = sum(rates) / len(rates) * 100
buckets = {"<30%": 0, "30-60%": 0, "60-90%": 0, "≥90%": 0}
for x in rates:
    if x < 0.3: buckets["<30%"] += 1
    elif x < 0.6: buckets["30-60%"] += 1
    elif x < 0.9: buckets["60-90%"] += 1
    else: buckets["≥90%"] += 1

# 按类别统计
from collections import defaultdict
by_cat = defaultdict(list)
for r in ROWS:
    rate, *_ = hit_rate(r)
    if rate is not None:
        by_cat[r["category"]].append(rate)
cat_stats = {k: sum(v)/len(v)*100 for k, v in by_cat.items()}

html_parts = ["""<!DOCTYPE html>
<html lang='zh'>
<head>
<meta charset='UTF-8'>
<title>v2 SFT 种子批 35 题推理结果</title>
<style>
body { font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 20px auto; padding: 20px; line-height: 1.5; }
.summary { background: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
.q { border: 1px solid #ddd; border-radius: 6px; padding: 14px; margin-bottom: 14px; }
.q.lo { border-left: 6px solid #d9534f; }
.q.mid { border-left: 6px solid #f0ad4e; }
.q.hi { border-left: 6px solid #5cb85c; }
.q.dual { border-left: 6px solid #5bc0de; }
.head { font-weight: bold; margin-bottom: 8px; }
.tag { display: inline-block; padding: 2px 6px; background: #eef; margin-right: 4px; font-size: 12px; border-radius: 3px; }
.kp-hit { color: #5cb85c; }
.kp-miss { color: #d9534f; font-weight: bold; }
.q-text { background: #f9f9f9; padding: 8px; border-radius: 4px; margin: 6px 0; }
.ref { background: #f0f8e8; padding: 8px; border-radius: 4px; margin: 6px 0; }
.ans { background: #fff5e6; padding: 8px; border-radius: 4px; margin: 6px 0; white-space: pre-wrap; font-size: 14px; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
th { background: #f5f5f5; }
</style>
</head>
<body>
<h1>v2 SFT 种子批推理结果（35 题）</h1>
"""]

html_parts.append(f"""<div class='summary'>
<b>整体命中率（基于参考答案 key_points 字面匹配）</b><br>
平均: {avg:.1f}% | 分布: <30%={buckets["<30%"]} 题 / 30-60%={buckets["30-60%"]} 题 / 60-90%={buckets["60-90%"]} 题 / ≥90%={buckets["≥90%"]} 题<br>
<br>
<table><tr><th>类别</th><th>平均命中率</th></tr>""")
for cat, rate in sorted(cat_stats.items(), key=lambda x: -x[1]):
    html_parts.append(f"<tr><td>{html.escape(cat)}</td><td>{rate:.1f}%</td></tr>")
html_parts.append("</table>")
html_parts.append("<p>注: 命中率仅是字面 key_point 匹配,不等于答案对错。需逐条人工评。</p></div>")

# 每条详细
for r in ROWS:
    rate, hits, miss = hit_rate(r)
    rate_pct = rate * 100 if rate is not None else 0
    # 颜色档:0-30 红 / 30-60 黄 / 60+ 绿,DUAL 蓝
    if r["system_track"] == "DUAL":
        cls = "dual"
    elif rate_pct < 30:
        cls = "lo"
    elif rate_pct < 60:
        cls = "mid"
    else:
        cls = "hi"
    html_parts.append(f"<div class='q {cls}'>")
    html_parts.append(f"<div class='head'>"
                      f"<span class='tag'>{r['id']}</span>"
                      f"<span class='tag'>{html.escape(r['category'])}</span>"
                      f"<span class='tag'>{r['difficulty']}</span>"
                      f"<span class='tag'>system={r['run_track']}</span>"
                      f"命中率: {rate_pct:.0f}% ({len(hits)}/{len(r['key_points'])})</div>")
    html_parts.append(f"<div class='q-text'><b>题目:</b> {html.escape(r['question'])}</div>")
    html_parts.append(f"<div class='ref'><b>参考答案:</b><br>{html.escape(r['reference_answer']).replace(chr(10), '<br>')}</div>")
    html_parts.append(f"<div class='ans'><b>模型答案:</b><br>{html.escape(r['model_answer']).replace(chr(10), '<br>')}</div>")
    if hits:
        html_parts.append(f"<div><b class='kp-hit'>✓ 命中:</b> {', '.join(html.escape(k) for k in hits)}</div>")
    if miss:
        html_parts.append(f"<div><b class='kp-miss'>✗ 缺失:</b> {', '.join(html.escape(k) for k in miss)}</div>")
    if r.get("notes"):
        html_parts.append(f"<div><i>📝 {html.escape(r['notes'])}</i></div>")
    html_parts.append("</div>")

html_parts.append("</body></html>")

out = r"C:\Users\PC\Claude脚本\medical_v2_eval\seed_report.html"
with open(out, "w", encoding="utf-8") as f:
    f.write("".join(html_parts))
print(f"wrote {out}")
print(f"avg hit rate = {avg:.1f}%")
print(f"buckets: {buckets}")
