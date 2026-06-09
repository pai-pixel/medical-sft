#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path

DEFAULT_SYSTEM_FALLBACK = "你是一位资深医师,熟悉临床诊断和健康咨询。"

USER_KEYS = ["query","question","instruction","prompt","input","user","Question","Instruction"]
ASSISTANT_KEYS = ["response","answer","output","target","assistant","Answer","Response","Output"]

NOISE_PATTERNS = [
    re.compile(r"汉药"), re.compile(r"东中业"),
    re.compile(r"作为人工智能"), re.compile(r"我是一个AI"), re.compile(r"我作为AI"),
]
LAZY_RE = re.compile(r"^(很抱歉|抱歉|很不好意思).{0,30}(无法|没有提供|没有提到|不能).{0,200}$")
QA_PREFIX = re.compile(r"^(问|答)[:：]\s*", re.MULTILINE)
MIN_LEN = 80


def is_noise(t):
    if len(t.strip()) < MIN_LEN: return True
    if LAZY_RE.match(t): return True
    for p in NOISE_PATTERNS:
        if p.search(t): return True
    return False


def pick(d, keys):
    for k in keys:
        v = d.get(k)
        if v is None: continue
        s = str(v).strip()
        if s: return s
    return None


def normalize_one(rec, system_text):
    # HuatuoGPT 格式: {"data": ["问:...", "答:..."]}
    if "data" in rec and isinstance(rec["data"], list) and len(rec["data"]) >= 2:
        u_raw = str(rec["data"][0]).strip()
        a_raw = str(rec["data"][1]).strip()
        u = QA_PREFIX.sub("", u_raw, count=1).strip()
        a = QA_PREFIX.sub("", a_raw, count=1).strip()
        if u and a:
            return {"system": system_text, "messages": [
                {"role":"user","content":u},
                {"role":"assistant","content":a}]}
        return None
    # 通用 instruction/input/output 或 query/response
    instr = (rec.get("instruction") or rec.get("Instruction") or "").strip()
    extra = (rec.get("input") or rec.get("Input") or "").strip()
    if instr and extra and instr.lower() != extra.lower():
        user = f"{instr}\n{extra}"
    else:
        user = pick(rec, USER_KEYS)
    asst = pick(rec, ASSISTANT_KEYS)
    if not user or not asst: return None
    return {"system": system_text, "messages": [
        {"role":"user","content":user},
        {"role":"assistant","content":asst}]}


def iter_records(path):
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if not first.strip(): return
        if first.lstrip().startswith("["):
            f.seek(0); arr = json.load(f)
            for r in arr: yield r
            return
        try: yield json.loads(first)
        except: return
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except: continue


def hash01(k):
    h = hashlib.sha256(k.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / (1<<64)


def process(src, default_system, out_path):
    name = src["name"]; path = Path(src["path"])
    quota = src.get("quota"); apply_filter = src.get("filter", False)
    track = src.get("track", "?"); src_system = src.get("system") or default_system
    print(f"[{time.strftime('%H:%M:%S')}] === {name} ({track}) ===  path={path}", flush=True)
    if not path.exists():
        print(f"  MISS", flush=True)
        return {"name":name,"kept":0,"missing":True}
    total = sum(1 for _ in iter_records(path))
    if quota and quota < total:
        thr = quota / total; print(f"  total={total} quota={quota} thr={thr:.4f}", flush=True)
    else:
        thr = None; print(f"  total={total} use_all", flush=True)
    kept=dropped=filtered=0
    with open(out_path, "a", encoding="utf-8") as fo:
        for i, rec in enumerate(iter_records(path)):
            if not isinstance(rec, dict):
                dropped += 1; continue
            if thr is not None and hash01(f"{name}::{i}") >= thr: continue
            norm = normalize_one(rec, src_system)
            if norm is None:
                dropped += 1; continue
            if apply_filter and is_noise(norm["messages"][1]["content"]):
                filtered += 1; continue
            norm["_source"] = name; norm["_track"] = track
            fo.write(json.dumps(norm, ensure_ascii=False) + "\n")
            kept += 1
    print(f"  kept={kept} dropped={dropped} filtered={filtered}", flush=True)
    return {"name":name,"track":track,"total":total,"kept":kept,"dropped":dropped,"filtered":filtered}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()
    summary = []
    for src in cfg["sources"]:
        if src.get("disabled"):
            print(f"--- skip {src['name']} ---"); continue
        summary.append(process(src, DEFAULT_SYSTEM_FALLBACK, out))
    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    by_track = {}; total = 0
    for s in summary:
        k = s.get("kept", 0); total += k
        t = s.get("track", "?"); by_track[t] = by_track.get(t, 0) + k
        print(f"  [{t:8s}] {s['name']:40s} kept={k:8d}")
    print()
    for t, c in by_track.items():
        print(f"  TRACK {t:8s}: {c:8d}  ({c/total*100:.1f}%)")
    print(f"  TOTAL = {total}  size_MB={out.stat().st_size/1024/1024:.1f}")


if __name__ == "__main__":
    main()
