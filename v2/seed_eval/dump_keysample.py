"""详细打印关键 8 条到 txt(utf-8 不再被 GBK 终端弄崩)。
重点关注: TCM 方剂回归 / TCM 中药低分 / EBM 拒答 / 双轨切换。"""
import json
ROWS = []
with open(r"C:\Users\PC\Claude脚本\medical_v2_eval\seed_inference_out.jsonl", encoding="utf-8") as f:
    for line in f:
        ROWS.append(json.loads(line))
ROW = {r["id"] + "|" + r["run_track"]: r for r in ROWS}

# 选关键 ID
PICKS = [
    ("S001|TCM", "六味地黄丸 (v1 回归)"),
    ("S003|TCM", "二陈汤 (v1 回归)"),
    ("S005|TCM", "理中丸 (v1 回归)"),
    ("S007|TCM", "附子配伍禁忌 (中药 0%)"),
    ("S008|TCM", "黄芪 (中药 0%)"),
    ("S024|EBM", "鸡汤治感冒 (拒答测试)"),
    ("S025|EBM", "薏米根治糖尿病 (拒答测试)"),
    ("S026|EBM", "失眠 - EBM 轨"),
    ("S026|TCM", "失眠 - TCM 轨"),
    ("S014|TCM", "胸痹 vs 真心痛 (难题)"),
    ("S019|EBM", "STEMI 早期处理"),
]

with open(r"C:\Users\PC\Claude脚本\medical_v2_eval\seed_keysample.txt", "w", encoding="utf-8") as f:
    for key, title in PICKS:
        r = ROW[key]
        f.write("=" * 100 + "\n")
        f.write(f"【{r['id']} | {title} | system={r['run_track']} | {r['difficulty']}】\n")
        f.write("=" * 100 + "\n")
        f.write(f"题目: {r['question']}\n\n")
        f.write(f"参考答案 ({len(r['reference_answer'])} 字):\n{r['reference_answer']}\n\n")
        f.write(f"模型答案 ({len(r['model_answer'])} 字):\n{r['model_answer']}\n\n")
        f.write(f"key_points 应命中: {r['key_points']}\n")
        f.write("\n\n")
print("wrote seed_keysample.txt")
