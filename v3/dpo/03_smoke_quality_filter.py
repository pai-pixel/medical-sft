"""Phase 1.2.5 smoke: Sonnet 4.6 给 v2 prompts 打质量分.
先跑 50 条 (5 batch x 10) 验证 JSON 解析 + 评分分布合理性.
"""
import os, json, asyncio, shutil, resource, collections
import aiohttp

# fd 抬到 65536
_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _h), _h))
print(f"[smoke] RLIMIT_NOFILE {_s} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}", flush=True)

# load key
api_key = None
with open("/mnt/data/huangjiawei/.config/dashscope.env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DASHSCOPE_API_KEY="):
            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
assert api_key and api_key.startswith("ak-"), "key load failed"

URL = "https://models-proxy.stepfun-inc.com/v1/chat/completions"
MODEL = "claude-sonnet-4-6:mindracode-anthropic-qianli"
BATCH = 10
SEM = 4  # smoke 用小并发

INP = "/mnt/data/huangjiawei/datasets_local/medical_dpo/prompts_v2_40k.jsonl"
LOCAL_INP = "/tmp/prompts_v2_40k.jsonl"
if not os.path.exists(LOCAL_INP):
    shutil.copy(INP, LOCAL_INP)

# 读前 50 条
prompts = []
with open(LOCAL_INP, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 50: break
        prompts.append(json.loads(line))
print(f"[smoke] loaded {len(prompts)} prompts", flush=True)

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
    items_str = "\n".join(
        f'{{"id": {i}, "prompt": {json.dumps(p["prompt"], ensure_ascii=False)}}}'
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
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": body}],
                        "temperature": 0.1,
                        "max_tokens": 1500,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        last_err = f"http_{resp.status}: {text[:200]}"
                        if "429" in text or "rate" in text.lower():
                            await asyncio.sleep(2 ** attempt); continue
                        return None
                    data = json.loads(text)
                    out = data["choices"][0]["message"]["content"].strip()
                    if out.startswith("```"):
                        out = out.split("```")[1]
                        if out.startswith("json"): out = out[4:]
                    out = out.strip()
                    parsed = json.loads(out)
                    return {"batch": batch_idx, "raw_out": out[:500], "parsed": parsed,
                            "items_count": len(items), "parsed_count": len(parsed)}
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:150]}"
                await asyncio.sleep(2 ** attempt)
        print(f"  [FAIL batch {batch_idx}] {last_err}", flush=True)
        return {"batch": batch_idx, "error": last_err}

async def main():
    batches = [prompts[i:i+BATCH] for i in range(0, len(prompts), BATCH)]
    print(f"[smoke] {len(batches)} batches", flush=True)
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(SEM)
        tasks = [score_batch(session, sem, i, b) for i, b in enumerate(batches)]
        results = []
        for fut in asyncio.as_completed(tasks):
            r = await fut
            if r: results.append(r)
    print(f"\n=== {len(results)} batches done ===", flush=True)

    ok_batches = [r for r in results if "parsed" in r]
    fail_batches = [r for r in results if "error" in r]

    print(f"\n=== batch status ===")
    print(f"  OK:   {len(ok_batches)}/{len(results)}")
    print(f"  FAIL: {len(fail_batches)}/{len(results)}")
    if fail_batches:
        for r in fail_batches:
            print(f"  - batch {r['batch']}: {r['error']}")

    print(f"\n=== parse alignment (parsed_count vs items_count) ===")
    for r in ok_batches:
        align = "OK" if r["parsed_count"] == r["items_count"] else "MISALIGN"
        print(f"  batch {r['batch']}: {r['parsed_count']}/{r['items_count']} [{align}]")

    # 收集所有 scores
    all_scores = []
    for r in ok_batches:
        for entry in r["parsed"]:
            sc = entry.get("score")
            if isinstance(sc, (int, float)):
                all_scores.append(int(sc))

    print(f"\n=== score distribution ({len(all_scores)} scored) ===")
    dist = collections.Counter(all_scores)
    for s in sorted(dist):
        bar = "#" * dist[s]
        print(f"  {s:2d}: {dist[s]:3d} {bar}")

    if all_scores:
        print(f"\n  min={min(all_scores)} max={max(all_scores)} mean={sum(all_scores)/len(all_scores):.2f}")

    # 抽 3 个 batch 的样本 (raw)
    print(f"\n=== 抽样 raw output (batch 0) ===")
    if ok_batches:
        print(ok_batches[0]["raw_out"][:800])

    # 抽几个高分/低分 prompt + 理由
    print(f"\n=== 高分样本 (score=10) ===")
    hi_count = 0
    for r in ok_batches:
        if hi_count >= 3: break
        for i, entry in enumerate(r["parsed"]):
            if entry.get("score") == 10 and hi_count < 3:
                # 找对应 prompt
                batch_items = prompts[r["batch"]*BATCH:(r["batch"]+1)*BATCH]
                if entry.get("id") < len(batch_items):
                    p = batch_items[entry["id"]]["prompt"]
                    print(f"  [{entry.get('reason','')}] {p[:120]}")
                    hi_count += 1

    print(f"\n=== 低分样本 (score<=4) ===")
    lo_count = 0
    for r in ok_batches:
        if lo_count >= 5: break
        for i, entry in enumerate(r["parsed"]):
            sc = entry.get("score", 5)
            if sc <= 4 and lo_count < 5:
                batch_items = prompts[r["batch"]*BATCH:(r["batch"]+1)*BATCH]
                if entry.get("id") < len(batch_items):
                    p = batch_items[entry["id"]]["prompt"]
                    print(f"  score={sc} [{entry.get('reason','')}] {p[:150]}")
                    lo_count += 1

asyncio.run(main())
