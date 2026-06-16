import os, json, random
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from datasets import load_dataset

OUT = "/mnt/data/huangjiawei/datasets_local/medical_mcq"
os.makedirs(OUT, exist_ok=True)
random.seed(0)

# CMMLU medical-relevant subjects (skip any that fail to load)
SUBJECTS = ["traditional_chinese_medicine", "clinical_knowledge", "college_medicine",
            "anatomy", "medical_genetics", "nutrition", "virology", "human_sexuality"]

PROMPT_TMPL = (
    "以下是一道医学单项选择题，请先简要分析，再作答。\n\n"
    "题目：{q}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "请在最后一行用「答案：X」给出唯一正确选项，X 为 A、B、C、D 之一。"
)

records = []
for subj in SUBJECTS:
    for split in ["test", "dev"]:
        try:
            ds = load_dataset("haonan-li/cmmlu", subj, split=split)
        except Exception as e:
            print("  skip %s/%s: %r" % (subj, split, repr(e)[:100]), flush=True)
            continue
        cols = ds.column_names
        if not {"Question", "A", "B", "C", "D", "Answer"}.issubset(set(cols)):
            print("!! UNEXPECTED schema for %s: %s -> abort" % (subj, cols), flush=True)
            raise SystemExit(1)
        for r in ds:
            ans = str(r["Answer"]).strip().upper()
            if ans not in ("A", "B", "C", "D"):
                continue
            prompt = PROMPT_TMPL.format(q=str(r["Question"]).strip(), a=r["A"], b=r["B"], c=r["C"], d=r["D"])
            records.append({"messages": [{"role": "user", "content": prompt}],
                            "solution": ans, "subject": subj})

# dedup by prompt, shuffle, split
seen, uniq = set(), []
for r in records:
    key = r["messages"][0]["content"]
    if key in seen:
        continue
    seen.add(key); uniq.append(r)
random.shuffle(uniq)

N_HELD = 250
held, train = uniq[:N_HELD], uniq[N_HELD:]

def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

dump(os.path.join(OUT, "train.jsonl"), train)
dump(os.path.join(OUT, "heldout.jsonl"), held)

from collections import Counter
print("TOTAL uniq:", len(uniq), " train:", len(train), " heldout:", len(held), flush=True)
print("answer dist (all):", dict(Counter(r["solution"] for r in uniq)), flush=True)
print("subject dist:", dict(Counter(r["subject"] for r in uniq)), flush=True)
print("--- sample train record ---", flush=True)
print(json.dumps(train[0], ensure_ascii=False, indent=2), flush=True)
print("WROTE ->", OUT, flush=True)
