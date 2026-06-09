#!/usr/bin/env python3
"""Normalize a TCM SFT dataset (ShenNong / SylvanL / ChatMed-TCM ...) to
the ms-swift `messages` schema.

Input: a .jsonl or .json (array) file where each record has user/assistant
       text under common field names (auto-sniffed).
Output: jsonl with {"system": ..., "messages": [user, assistant]}.

Usage:
    python3 normalize_to_swift.py INPUT.jsonl OUTPUT.jsonl
    python3 normalize_to_swift.py INPUT.jsonl smoke.jsonl --max-samples 1000
"""

import argparse
import json
import sys
from pathlib import Path

USER_KEYS = [
    "query", "question", "instruction", "prompt", "input", "user",
    "Question", "Instruction",
]
ASSISTANT_KEYS = [
    "response", "answer", "output", "target", "assistant",
    "Answer", "Response", "Output",
]
SYSTEM_KEYS = ["system", "system_prompt", "System"]

DEFAULT_SYSTEM = (
    "你是一位资深的中医师，精通中医辨证论治、方剂、本草和经典医籍。"
    "请根据用户提问给出严谨、专业、易懂的中医解答。"
)


def pick(d, candidates):
    for k in candidates:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def normalize_one(record, default_system):
    instr = (record.get("instruction") or record.get("Instruction") or "").strip()
    extra = (record.get("input") or record.get("Input") or "").strip()
    if instr and extra and instr.lower() != extra.lower():
        user = f"{instr}\n{extra}"
    else:
        user = pick(record, USER_KEYS)
    asst = pick(record, ASSISTANT_KEYS)
    sys_p = pick(record, SYSTEM_KEYS) or default_system

    if not user or not asst:
        return None

    return {
        "system": sys_p,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst},
        ],
    }


def iter_records(path: Path):
    with path.open(encoding="utf-8") as f:
        first = f.readline()
        if not first.strip():
            return
        stripped = first.lstrip()
        if stripped.startswith("["):
            f.seek(0)
            arr = json.load(f)
            for r in arr:
                yield r
        else:
            yield json.loads(first)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input .jsonl or .json file")
    ap.add_argument("output", help="output .jsonl in swift format")
    ap.add_argument("--system", default=DEFAULT_SYSTEM,
                    help="default system prompt if record has none")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap number of output records (for smoke run)")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        sys.exit(1)
    outp.parent.mkdir(parents=True, exist_ok=True)

    total = kept = dropped = 0
    with outp.open("w", encoding="utf-8") as fo:
        for r in iter_records(inp):
            total += 1
            if not isinstance(r, dict):
                dropped += 1
                continue
            norm = normalize_one(r, args.system)
            if norm is None:
                dropped += 1
                continue
            fo.write(json.dumps(norm, ensure_ascii=False) + "\n")
            kept += 1
            if args.max_samples and kept >= args.max_samples:
                break

    print(f"input  : {inp}")
    print(f"output : {outp}")
    print(f"total  : {total}")
    print(f"kept   : {kept}")
    print(f"dropped: {dropped} (missing user/assistant text)")


if __name__ == "__main__":
    main()
