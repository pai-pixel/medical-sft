"""快速扫一遍所有答案,做 key_points 命中统计 + 输出对照报告。
不是最终评分(我后面会逐条评),先给一个 quick scan。"""
import json
import re

ROWS = []
with open(r"C:\Users\PC\Claude脚本\medical_v2_eval\seed_inference_out.jsonl", encoding="utf-8") as f:
    for line in f:
        ROWS.append(json.loads(line))

# 简单 key_points 命中: 题干里 key_point 文本是否出现在 model_answer 中
def hit_rate(row):
    ans = row["model_answer"]
    kps = row["key_points"]
    if not kps:
        return None, []
    hits = []
    miss = []
    for kp in kps:
        # 处理"山茱萸（山萸肉）"这种形式 — 任一别名命中算对
        alts = re.split(r"[（）()]", kp)
        alts = [a.strip() for a in alts if a.strip()]
        if any(a in ans for a in alts):
            hits.append(kp)
        else:
            miss.append(kp)
    return len(hits) / len(kps), miss

print("=" * 100)
print(f"{'ID':6} {'类别':16} {'难度':14} {'轨':4} {'命中率':8} {'缺失关键点'}")
print("=" * 100)
for r in ROWS:
    rate, miss = hit_rate(r)
    rate_s = f"{rate*100:.0f}%" if rate is not None else "N/A"
    print(f"{r['id']:6} {r['category']:16} {r['difficulty']:14} {r['run_track']:4} {rate_s:8} {','.join(miss[:5])}")

# overall
print()
print("=== 整体命中率分布 ===")
rates = [hit_rate(r)[0] for r in ROWS if hit_rate(r)[0] is not None]
buckets = {"<30%": 0, "30-60%": 0, "60-90%": 0, ">=90%": 0}
for x in rates:
    if x < 0.3:
        buckets["<30%"] += 1
    elif x < 0.6:
        buckets["30-60%"] += 1
    elif x < 0.9:
        buckets["60-90%"] += 1
    else:
        buckets[">=90%"] += 1
print(buckets)
print(f"平均命中率: {sum(rates)/len(rates)*100:.1f}%")

# 长度分布
print()
print("=== 答案长度 ===")
lens = [(r["id"], len(r["model_answer"])) for r in ROWS]
short = sorted(lens, key=lambda x: x[1])[:5]
print("最短 5 条:", short)
