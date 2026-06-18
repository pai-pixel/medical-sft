"""Phase 1.3 smoke: Opus 4.7 生成 10 条 General prompt 验证流程.
- meta prompt 设计验证
- JSON 解析鲁棒性
- prompt 多样性 + 真实性观察
"""
import os, json, asyncio, resource, random
import aiohttp

_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _h), _h))

api_key = None
with open("/mnt/data/huangjiawei/.config/dashscope.env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DASHSCOPE_API_KEY="):
            api_key = line.split("=", 1)[1].strip().strip('"').strip("'"); break
assert api_key

URL = "https://models-proxy.stepfun-inc.com/v1/chat/completions"
MODEL = "claude-opus-4-7:mindracode-anthropic-qianli"
PER_CALL = 10
SMOKE_CALLS = 5  # 5 调用 = 50 条 prompt,看多样性

TOPICS = {
    "symptom": ["感冒发烧咳嗽", "头痛头晕失眠", "腰背肩颈疼痛",
                "胃疼腹泻便秘", "皮肤湿疹痤疮瘙痒", "心慌心悸胸闷"],
    "safety": ["民间偏方求证", "用药过量误服", "中毒应对",
               "意外伤害急救", "孕妇用药安全", "婴幼儿用药"],
    "refuse": ["想自己治某病的危险问法", "求大量处方药剂量",
               "想绕过医生开药", "自我伤害倾向", "未经验证偏方"],
    "family": ["老人慢病护理", "孩子发烧喂养", "癌症患者陪护",
               "孕妇日常注意", "家人意外昏迷"],
    "ambig_tcm_vs_ebm": ["中医说阴虚西医查无异常", "中药 vs 西药选择",
                          "中医穴位 vs 西医靶点", "中西医结合治癌"],
    "ambig_rare": ["罕见病初步排查", "复合症状无明确诊断",
                    "查不出原因的不适", "多系统疾病困惑"],
}
CATEGORY_DESC = {
    "symptom": "用户描述症状求初判,通常带情绪",
    "safety": "涉及安全边界,需谨慎引导,不能直接开药",
    "refuse": "危险/需拒答场景,模型应拒答+引导就医",
    "family": "用户问家人(老人/孩子/孕妇/患者)健康",
    "ambig_tcm_vs_ebm": "中西医意见冲突或需要兼顾",
    "ambig_rare": "罕见病/复合症状/诊断不明",
}

META = """你是医疗对话数据集设计师。请生成 {n} 个真实用户向 AI 医疗助手提出的问题。

种子主题(本次方向, 随意发挥):
{seeds}

要求:
1. 真实性: 真用户口吻(口语化, 可不完整, 可带情绪/焦虑/不耐烦)
2. 多样性: 每条独立, 不要模板复用, 句式句长都要变化
3. 长度: 每条 15-200 字
4. 内容范围: {desc}
5. 现实痛点: 模拟真实健康困惑, 不要太"完美"或"教科书式"

输出严格 JSON list, {n} 项, 每项 {{"prompt": "...", "type": "{ct}"}}.
直接 [ 开始, 无 markdown 标记, 无前后解释。"""

async def gen_one(session, sem, call_idx, category):
    seeds = random.sample(TOPICS[category], min(3, len(TOPICS[category])))
    seeds_str = "\n".join(f"- {s}" for s in seeds)
    body_prompt = META.format(n=PER_CALL, seeds=seeds_str,
                               desc=CATEGORY_DESC[category], ct=category)
    last_err = "no_exception"
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(
                    URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": MODEL,
                          "messages": [{"role": "user", "content": body_prompt}],
                          "temperature": 0.95, "max_tokens": 2000},
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        last_err = f"http_{resp.status}: {text[:150]}"
                        if "429" in text or "rate" in text.lower():
                            await asyncio.sleep(2 ** attempt); continue
                        return {"_fail": True, "_err": last_err, "_cat": category}
                    data = json.loads(text)
                    out = data["choices"][0]["message"]["content"].strip()
                    if out.startswith("```"):
                        out = out.split("```")[1]
                        if out.startswith("json"): out = out[4:]
                    out = out.strip()
                    parsed = json.loads(out)
                    return {"call": call_idx, "category": category,
                            "raw": out[:500], "parsed": parsed,
                            "count": len(parsed)}
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:100]}"
                await asyncio.sleep(2 ** attempt)
        return {"_fail": True, "_err": last_err, "_cat": category}

async def main():
    random.seed(7)
    categories = list(TOPICS.keys())
    calls = [(i, random.choice(categories)) for i in range(SMOKE_CALLS)]
    print(f"[smoke] {SMOKE_CALLS} calls, categories: {[c for _,c in calls]}", flush=True)

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(3)
        tasks = [gen_one(session, sem, i, c) for i, c in calls]
        results = [await fut for fut in asyncio.as_completed(tasks)]

    ok = [r for r in results if "parsed" in r]
    fail = [r for r in results if r.get("_fail")]
    print(f"\n=== {len(ok)}/{SMOKE_CALLS} ok, {len(fail)} fail ===", flush=True)
    if fail:
        for r in fail:
            print(f"  - {r.get('_cat')}: {r.get('_err')}")

    print(f"\n=== parse alignment ===")
    for r in ok:
        align = "OK" if r["count"] == PER_CALL else "MISALIGN"
        print(f"  call {r['call']} [{r['category']}]: {r['count']}/{PER_CALL} [{align}]")

    # 抽 1 个 batch raw 看格式
    print(f"\n=== raw output (call 0) ===")
    if ok:
        print(ok[0]["raw"][:1000])

    # 各 category 抽 3 个 prompt 样本
    print(f"\n=== 样本 (按 category) ===")
    by_cat = {}
    for r in ok:
        by_cat.setdefault(r["category"], []).extend(r["parsed"])
    for cat, items in by_cat.items():
        print(f"\n--- {cat} ({len(items)} 条) ---")
        for it in items[:3]:
            p = it.get("prompt", "")
            print(f"  · {p[:160]}")

    # 长度分布
    all_lens = [len(it.get("prompt", "")) for r in ok for it in r["parsed"]]
    if all_lens:
        all_lens.sort()
        print(f"\n=== 长度分布 ===")
        print(f"  min={all_lens[0]} p10={all_lens[len(all_lens)//10]} "
              f"p50={all_lens[len(all_lens)//2]} p90={all_lens[len(all_lens)*9//10]} "
              f"max={all_lens[-1]} mean={sum(all_lens)/len(all_lens):.0f}")

asyncio.run(main())
