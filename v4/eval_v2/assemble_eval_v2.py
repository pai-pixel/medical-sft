"""扩 v4 评估集 v2: 1390 题 (280 CMB + 1000 MedQA + 60 + 50)
跟原 v3 评估集兼容, 新增 CMB 全量 + MedQA-CN 1000
"""
import json, csv, re, random, itertools
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)

OUT = Path("/mnt/data/huangjiawei/datasets_local/eval/eval_v2_1390.jsonl")

records = []

# 1. v2_seed 30 (原)
with open("/mnt/data/huangjiawei/datasets_local/eval/seed_questions.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        x = json.loads(line)
        records.append({
            "id": f"seed_{x['id']}",
            "source": "v2_seed",
            "category": x.get("category", ""),
            "type": "open",
            "question": x["question"],
            "reference_answer": x["reference_answer"],
            "key_points": x["key_points"],
            "difficulty": x.get("difficulty", ""),
            "system_track": x.get("system_track", ""),
        })
print(f"[seed] {len(records)}")

# 2. new_30
n0 = len(records)
with open("/mnt/data/huangjiawei/datasets_local/eval/new_30_questions.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        x = json.loads(line)
        records.append({
            "id": f"new_{x['id']}",
            "source": "new_30",
            "category": x.get("category", ""),
            "type": "open",
            "question": x["question"],
            "reference_answer": x["reference_answer"],
            "key_points": x["key_points"],
            "difficulty": x.get("difficulty", ""),
            "system_track": x.get("system_track", ""),
        })
print(f"[new_30] {len(records)-n0}")

# 3. CMB 全量 (280 中只取单选, 排除考研政治)
n0 = len(records)
with open("/mnt/data/huangjiawei/datasets_local/eval/CMB/CMB-Exam/CMB-val/CMB-val-merge.json", encoding="utf-8") as f:
    cmb_all = json.load(f)
cmb_filt = [
    x for x in cmb_all
    if len(x["answer"]) == 1
    and x["exam_class"] != "考研政治"
    and isinstance(x.get("option"), dict)
    and x["answer"] in x["option"]
]
random.shuffle(cmb_filt)
for i, x in enumerate(cmb_filt):
    records.append({
        "id": f"cmb_{i:03d}",
        "source": "CMB",
        "category": f"CMB_{x['exam_class']}",
        "type": "mcq",
        "question": x["question"],
        "options": x["option"],
        "answer": x["answer"],
        "explanation": x.get("explanation", ""),
        "difficulty": "intermediate",
    })
print(f"[CMB] +{len(records)-n0}")

# 4. MedQA-CN 1000 (从 test 3426 抽 1000)
n0 = len(records)
medqa_all = []
with open("/mnt/data/huangjiawei/datasets_local/eval/MedQA/Mainland_questions/test.jsonl", encoding="utf-8") as f:
    for line in itertools.islice(f, 4000):
        x = json.loads(line)
        # 必须 5 选项 + answer_idx 在 ABCDE
        if x.get("answer_idx") in "ABCDE" and len(x.get("options", {})) >= 4:
            medqa_all.append(x)
random.shuffle(medqa_all)
medqa_sample = medqa_all[:1000]
for i, x in enumerate(medqa_sample):
    records.append({
        "id": f"medqa_{i:04d}",
        "source": "MedQA",
        "category": f"MedQA_{x.get('meta_info', '其他')}",
        "type": "mcq",
        "question": x["question"],
        "options": x["options"],
        "answer": x["answer_idx"],
        "explanation": "",
        "difficulty": "intermediate",
    })
print(f"[MedQA] +{len(records)-n0}")

# 5. C-Eval 50 (沿用原 assemble 的子集)
n0 = len(records)
import pandas as pd
CEVAL_BASE = Path("/mnt/data/huangjiawei/datasets_local/eval/C-Eval")
non_med = [
    "chinese_language_and_literature", "art_studies", "civil_servant",
    "modern_chinese_history", "ideological_and_moral_cultivation",
    "logic", "law", "professional_tour_guide",
    "education_science", "high_school_history",
    "high_school_geography", "middle_school_history",
    "middle_school_geography", "high_school_chinese",
    "marxism", "mao_zedong_thought",
]
ceval_pool = []
for sub in non_med:
    fp = CEVAL_BASE / sub / "val-00000-of-00001.parquet"
    if not fp.exists(): continue
    df = pd.read_parquet(fp)
    for _, row in df.iterrows():
        if pd.isna(row.get("answer")) or row["answer"] not in "ABCD": continue
        ceval_pool.append({
            "subject": sub,
            "question": row["question"],
            "options": {k: row[k] for k in "ABCD"},
            "answer": row["answer"],
        })
random.shuffle(ceval_pool)
for i, x in enumerate(ceval_pool[:50]):
    records.append({
        "id": f"ceval_{i:03d}",
        "source": "C-Eval",
        "category": f"C-Eval_{x['subject']}",
        "type": "mcq",
        "question": x["question"],
        "options": x["options"],
        "answer": x["answer"],
        "explanation": "",
        "difficulty": "basic",
    })
print(f"[C-Eval] +{len(records)-n0}")

# 写
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n[total] {len(records)}")
print("[by_source]:", dict(Counter(r["source"] for r in records)))
print("[by_type]:", dict(Counter(r["type"] for r in records)))
print(f"\n[saved] {OUT}")
