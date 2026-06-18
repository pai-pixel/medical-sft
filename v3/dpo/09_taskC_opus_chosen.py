"""Task C (Phase 2.3): Opus 4.7 生成 EBM 18K + General 7.5K = 25.5K chosen.
- 严格 domain-aware: 只跑 EBM/General, TCM 由 M2 跑
- staging 持久化 + resume
- SEM=8 + chunked wave + retry 4 次
- 系统 prompt 引导: 循证答 / 安全边界 / 拒答场景

预计 6-14 小时 (取决于 Opus 网关实际 RPM).
"""
import os, json, asyncio, hashlib, resource, shutil, time
import aiohttp
from collections import Counter

_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _h), _h))
print(f"[task-c] fd: {_s} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}", flush=True)

api_key = None
with open("/mnt/data/huangjiawei/.config/dashscope.env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DASHSCOPE_API_KEY="):
            api_key = line.split("=", 1)[1].strip().strip('"').strip("'"); break
assert api_key

URL = "https://models-proxy.stepfun-inc.com/v1/chat/completions"
MODEL = "claude-opus-4-7:mindracode-anthropic-qianli"
SEM = 8
WAVE = 200

OUT_DIR = "/mnt/data/huangjiawei/datasets_local/medical_dpo"
INP = os.path.join(OUT_DIR, "prompts_pool_43k.jsonl")
STAGING = os.path.join(OUT_DIR, "chosen_opus.jsonl.staging")
OUT = os.path.join(OUT_DIR, "chosen_opus_25k.jsonl")

# 加载 prompts (只 EBM + General)
prompts = []
with open(INP, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["domain"] in ("EBM", "General"):
            prompts.append(r)
print(f"[task-c] to-do: {len(prompts)} (EBM + General)", flush=True)

# resume
done = set()
if os.path.exists(STAGING):
    with open(STAGING, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(r["id"])
            except Exception:
                continue
print(f"[task-c] resume: {len(done)} already done", flush=True)
todo = [p for p in prompts if p["id"] not in done]
print(f"[task-c] todo after resume: {len(todo)}", flush=True)

SYS = """你是一位严谨的医学专家。回答用户医疗问题,要求:

1. 直接回答,不要客套前言
2. 西医循证: 引用具体指南 (IDSA / 中国 CAP / NICE) 或 RCT (具体研究名+发表年)
3. 中医辨证: 给证型+治法+代表方剂+加减化裁
4. 拒答场景 (自我伤害/危险用药/求绕过医生): 明确拒答 + 红旗症状清单 + 引导具体科室就医
5. 家人照护: 给具体行动清单,温和但专业
6. 用药咨询: 给具体药名+剂量,但提醒以医生处方为准 (不能给完整处方)
7. 复杂或不确定: 老实说"需进一步检查",列出可能的鉴别诊断和该挂的科室
8. 中文,不超过 1200 字,自然段落不用 markdown 标题"""

async def gen_one(session, sem, item):
    last_err = "no_exception"
    async with sem:
        for attempt in range(4):
            try:
                async with session.post(
                    URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": SYS},
                            {"role": "user", "content": item["prompt"]},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2500,
                    },
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        last_err = f"http_{resp.status}: {text[:100]}"
                        if resp.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(2 ** attempt); continue
                        return None
                    data = json.loads(text)
                    answer = data["choices"][0]["message"]["content"].strip()
                    if len(answer) < 30:
                        last_err = f"answer_too_short: {len(answer)}"
                        await asyncio.sleep(2 ** attempt); continue
                    return {
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "domain": item["domain"],
                        "chosen": answer,
                        "teacher": "opus-4-7",
                    }
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:80]}"
                await asyncio.sleep(2 ** attempt)
        print(f"  [FAIL id={item['id']}] {last_err}", flush=True)
        return None

async def main():
    if not todo:
        print("[task-c] all done, finalize only", flush=True)
    else:
        fout = open(STAGING, "a", encoding="utf-8")
        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(SEM)
            for wi in range(0, len(todo), WAVE):
                wave = todo[wi:wi+WAVE]
                tasks = [gen_one(session, sem, p) for p in wave]
                ok = 0
                for fut in asyncio.as_completed(tasks):
                    r = await fut
                    if r:
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                        ok += 1
                fout.flush()
                el = time.time() - t0
                done_now = wi + len(wave)
                eta = el / done_now * (len(todo) - done_now) if done_now > 0 else 0
                print(f"  wave {wi}/{len(todo)}: +{ok}/{len(wave)}, "
                      f"el={el/60:.1f}m, ETA={eta/60:.0f}m", flush=True)
        fout.close()
        print(f"[task-c] gen done in {(time.time()-t0)/60:.1f}m", flush=True)

    # finalize
    results = []
    seen = set()
    with open(STAGING, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r["id"] not in seen:
                    seen.add(r["id"])
                    results.append(r)
            except Exception:
                continue

    tmp_out = "/tmp/chosen_opus_final.jsonl"
    with open(tmp_out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(tmp_out, OUT)

    ddist = Counter(r["domain"] for r in results)
    print(f"\n=== final ===", flush=True)
    print(f"  total: {len(results)}/{len(prompts)}")
    for d, n in ddist.most_common():
        print(f"  {d}: {n}")
    print(f"  wrote -> {OUT}")

asyncio.run(main())
