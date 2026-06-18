"""Phase 1.2.5 full: Sonnet 4.6 给 v2 40K prompts 打质量分, 过滤 bottom 10%.
预计 25-35 分钟 (40K / 10 per batch / SEM 8 / Sonnet 2.96s 平均).
- staging 持久化防中途崩 + resume 支持
- chunked 200 batch / wave 防 event loop 撑爆
- 失败 3 次重试, 永久失败 fallback score=5 (中位, 不影响过滤)
"""
import os, json, asyncio, shutil, resource, collections, time
import aiohttp

_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _h), _h))
print(f"[full] RLIMIT_NOFILE {_s} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}", flush=True)

api_key = None
with open("/mnt/data/huangjiawei/.config/dashscope.env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DASHSCOPE_API_KEY="):
            api_key = line.split("=", 1)[1].strip().strip('"').strip("'"); break
assert api_key and api_key.startswith("ak-")

URL = "https://models-proxy.stepfun-inc.com/v1/chat/completions"
MODEL = "claude-sonnet-4-6:mindracode-anthropic-qianli"
BATCH = 10
SEM = 8
WAVE = 200  # async chunk size

OUT_DIR = "/mnt/data/huangjiawei/datasets_local/medical_dpo"
INP = os.path.join(OUT_DIR, "prompts_v2_40k.jsonl")
OUT = os.path.join(OUT_DIR, "prompts_v2_36k_filtered.jsonl")
STAGING = os.path.join(OUT_DIR, "prompts_v2_scores.jsonl.staging")
LOCAL_INP = "/tmp/prompts_v2_40k.jsonl"

if not os.path.exists(LOCAL_INP):
    shutil.copy(INP, LOCAL_INP)

prompts = []
with open(LOCAL_INP, encoding="utf-8") as f:
    for line in f:
        prompts.append(json.loads(line))
print(f"[full] loaded {len(prompts)} prompts", flush=True)

# resume
done_indices = set()
if os.path.exists(STAGING):
    with open(STAGING, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done_indices.add(r["idx"])
            except Exception:
                continue
print(f"[full] resume: {len(done_indices)} already scored", flush=True)

PROMPT_TPL = """以下是 {n} 条用户向医疗 AI 助手的提问。请逐条打 1-10 分,基于:
- 完整性: 信息够回答(模糊问题低分,如"我难受")
- 真实性: 像真用户口吻(口语化 OK)
- 医疗相关: 真实医疗咨询(无关问题低分)
- 隐私安全: 含具体姓名/电话/身份证给低分
- 学习价值: 学生模型答能学到东西

输出严格 JSON list, {n} 项, 每项 {{"id": 数字, "score": 1-10, "reason": "<15字理由"}}.

prompts:
{items}

只输出 JSON list (从 [ 开始 ], 无 markdown 标记, 无前后解释。"""

async def score_batch(session, sem, batch_idx, items):
    # items: list of (global_idx, prompt_dict)
    items_str = "\n".join(
        f'{{"id": {i}, "prompt": {json.dumps(p[1]["prompt"], ensure_ascii=False)}}}'
        for i, p in enumerate(items)
    )
    body = PROMPT_TPL.format(n=len(items), items=items_str)
    last_err = "no_exception"
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(
                    URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": MODEL,
                          "messages": [{"role": "user", "content": body}],
                          "temperature": 0.1, "max_tokens": 1500},
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        last_err = f"http_{resp.status}: {text[:150]}"
                        if "429" in text or "rate" in text.lower():
                            await asyncio.sleep(2 ** attempt); continue
                        if "403" in str(resp.status):
                            return None
                        await asyncio.sleep(2 ** attempt); continue
                    data = json.loads(text)
                    out = data["choices"][0]["message"]["content"].strip()
                    if out.startswith("```"):
                        out = out.split("```")[1]
                        if out.startswith("json"): out = out[4:]
                    out = out.strip()
                    parsed = json.loads(out)
                    results = []
                    for entry in parsed:
                        lid = entry.get("id")
                        if not isinstance(lid, int) or lid >= len(items): continue
                        gi = items[lid][0]
                        sc = entry.get("score", 5)
                        if not isinstance(sc, (int, float)): sc = 5
                        results.append({"idx": gi, "score": int(sc),
                                        "reason": str(entry.get("reason", ""))[:30]})
                    return results
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:100]}"
                await asyncio.sleep(2 ** attempt)
        return {"_fail": True, "_batch": batch_idx, "_err": last_err,
                "_indices": [it[0] for it in items]}

async def main():
    todo = [(i, prompts[i]) for i in range(len(prompts)) if i not in done_indices]
    print(f"[full] todo: {len(todo)}", flush=True)
    if not todo:
        print("[full] all scored, skip to filter", flush=True)
    else:
        batches = []
        for i in range(0, len(todo), BATCH):
            batches.append(todo[i:i+BATCH])
        print(f"[full] {len(batches)} batches, ~{len(batches)*3/SEM/60:.1f} min", flush=True)

        fout = open(STAGING, "a", encoding="utf-8")
        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(SEM)
            for wi in range(0, len(batches), WAVE):
                wave = batches[wi:wi+WAVE]
                tasks = [score_batch(session, sem, wi+i, b) for i, b in enumerate(wave)]
                ok_n = fail_n = 0
                for fut in asyncio.as_completed(tasks):
                    r = await fut
                    if r is None:
                        fail_n += 1; continue
                    if isinstance(r, dict) and r.get("_fail"):
                        fail_n += 1
                        # 永久失败的 idx 填 score=5 兜底,不阻塞
                        for gi in r["_indices"]:
                            fout.write(json.dumps({"idx": gi, "score": 5,
                                                    "reason": "FAIL_FALLBACK"},
                                                  ensure_ascii=False) + "\n")
                        continue
                    for item in r:
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    ok_n += len(r)
                fout.flush()
                elapsed = time.time() - t0
                done_so_far = wi + len(wave)
                eta = elapsed / done_so_far * (len(batches) - done_so_far)
                print(f"  wave {wi}/{len(batches)}: ok={ok_n} fail={fail_n} "
                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
        fout.close()
        print(f"[full] scoring done in {time.time()-t0:.0f}s", flush=True)

    # 应用过滤
    scores = {}
    with open(STAGING, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r["idx"] not in scores or scores[r["idx"]] == 5:
                    scores[r["idx"]] = r["score"]
            except Exception:
                continue
    print(f"[full] loaded {len(scores)} scores from staging", flush=True)

    scored = []
    for i, p in enumerate(prompts):
        scored.append({**p, "_score": scores.get(i, 5)})

    dist = collections.Counter(p["_score"] for p in scored)
    print(f"\n=== final score distribution ===")
    for s in sorted(dist):
        bar = "#" * (dist[s] // 200)
        print(f"  {s:2d}: {dist[s]:6d} {bar}")
    print(f"  mean={sum(p['_score'] for p in scored)/len(scored):.2f}")

    # bottom 10% 过滤
    scored.sort(key=lambda x: x["_score"])
    drop_n = int(len(scored) * 0.10)
    dropped = scored[:drop_n]
    kept = scored[drop_n:]
    drop_threshold = scored[drop_n-1]["_score"] if drop_n > 0 else 0
    print(f"\n=== filter ===")
    print(f"  dropped bottom 10%: {len(dropped)} (score <= {drop_threshold})")
    print(f"  kept top 90%: {len(kept)}")

    # 写盘
    tmp_out = "/tmp/prompts_v2_36k_filtered.jsonl"
    import random; random.seed(42); random.shuffle(kept)
    with open(tmp_out, "w", encoding="utf-8") as f:
        for r in kept:
            score = r.pop("_score")
            r["quality_score"] = score
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(tmp_out, OUT)
    print(f"\nwrote -> {OUT}", flush=True)

    # kept 分布
    dom_dist = collections.Counter(r["domain"] for r in kept)
    print(f"\n=== kept domain dist ===")
    for k, v in dom_dist.most_common():
        print(f"  {k}: {v}")

asyncio.run(main())
