"""Phase 1.3 full: Opus 4.7 生成 10K General + Ambig prompts.
- 7 categories: General 7.5K (symptom+safety+refuse+family+drug) + Ambig 2.5K
- staging 持久化 + resume
- 每次 10 条, SEM=6, 失败 4 次重试 (含 502/503/429)
- 长度过滤 15-250 字, content hash 去重
预计 ~25-30 min.
"""
import os, json, asyncio, resource, random, hashlib, time, shutil
import aiohttp
from collections import defaultdict, Counter

_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _h), _h))
print(f"[full] fd: {_s} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}", flush=True)

api_key = None
with open("/mnt/data/huangjiawei/.config/dashscope.env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DASHSCOPE_API_KEY="):
            api_key = line.split("=", 1)[1].strip().strip('"').strip("'"); break

URL = "https://models-proxy.stepfun-inc.com/v1/chat/completions"
MODEL = "claude-opus-4-7:mindracode-anthropic-qianli"
PER_CALL = 10
SEM = 6
WAVE = 100

OUT_DIR = "/mnt/data/huangjiawei/datasets_local/medical_dpo"
OUT = os.path.join(OUT_DIR, "prompts_opus_gen_10k.jsonl")
STAGING = os.path.join(OUT_DIR, "prompts_opus_gen.jsonl.staging")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = {
    "symptom": 2000, "safety": 1500, "refuse": 1000,
    "family": 1000, "drug": 2000,
    "ambig_tcm_vs_ebm": 1500, "ambig_rare": 1000,
}
DOMAIN = {"symptom": "General", "safety": "General", "refuse": "General",
          "family": "General", "drug": "General",
          "ambig_tcm_vs_ebm": "Ambig", "ambig_rare": "Ambig"}

TOPICS = {
    "symptom": ["感冒发烧咳嗽", "头痛头晕失眠", "腰背肩颈疼痛", "胃疼腹泻便秘",
                "皮肤湿疹痤疮瘙痒", "心慌心悸胸闷", "眼花视力突变", "耳鸣听力下降",
                "口腔牙龈舌头问题", "尿频尿急血尿"],
    "drug": ["处方药使用咨询", "中西药同服安全", "保健品功效求证",
             "进口药与国产差异", "孕产妇用药安全", "婴幼儿用药剂量",
             "降压药用药管理", "降糖药用药管理", "抗生素何时该用",
             "中药煎服方法", "外用药使用"],
    "safety": ["民间偏方求证", "用药过量误服", "中毒应对", "意外伤害急救",
               "孕妇用药安全", "婴幼儿误吞", "老人误吞药物",
               "酒精中毒急救", "煤气中毒抢救", "烫伤烧伤紧急"],
    "refuse": ["想自己治某病的危险问法", "求大量处方药剂量",
               "想绕过医生开药", "自我伤害倾向", "未经验证偏方",
               "替家人买管制药", "求中止妊娠土方法", "求轻生方法"],
    "family": ["老人慢病护理", "孩子发烧喂养", "癌症患者陪护",
               "孕妇日常注意", "家人意外昏迷", "婴儿哭闹睡眠",
               "家有抑郁症患者", "老年痴呆照护", "术后恢复期"],
    "ambig_tcm_vs_ebm": ["中医说阴虚西医查无异常", "中药 vs 西药选择",
                          "中医穴位 vs 西医靶点", "中西医结合治癌",
                          "中医说湿气重西医不认", "中医辨证 vs 西医诊断"],
    "ambig_rare": ["罕见病初步排查", "复合症状无明确诊断",
                    "查不出原因的不适", "多系统疾病困惑",
                    "查了多个科都不确定", "症状像很多种病"],
}
DESC = {
    "symptom": "用户描述症状求初判,通常带情绪",
    "drug": "对药物使用/安全/剂量的咨询",
    "safety": "涉及安全边界,需谨慎引导",
    "refuse": "危险/需拒答场景,模型应拒答+引导就医",
    "family": "用户问家人(老人/孩子/孕妇/患者)健康",
    "ambig_tcm_vs_ebm": "中西医意见冲突或需要兼顾",
    "ambig_rare": "罕见病/复合症状/诊断不明",
}

META = """你是医疗对话数据集设计师。请生成 {n} 个真实用户向 AI 医疗助手提出的问题。

种子主题(本次方向, 可自由发挥):
{seeds}

要求:
1. 真实性: 真用户口吻(口语化, 可不完整, 带情绪/焦虑/不耐烦)
2. 多样性: 每条独立, 句式句长变化, 不要模板复用
3. 长度: 每条 15-200 字
4. 内容范围: {desc}
5. 现实痛点: 模拟真实健康困惑

输出严格 JSON list, {n} 项, 每项 {{"prompt": "...", "type": "{ct}"}}.
直接 [ 开始, 无 markdown 标记, 无前后解释。"""

async def gen_one(session, sem, call_idx, category):
    seeds = random.sample(TOPICS[category], min(3, len(TOPICS[category])))
    seeds_str = "\n".join(f"- {s}" for s in seeds)
    body = META.format(n=PER_CALL, seeds=seeds_str,
                       desc=DESC[category], ct=category)
    last_err = "no_exception"
    async with sem:
        for attempt in range(4):
            try:
                async with session.post(
                    URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": MODEL,
                          "messages": [{"role": "user", "content": body}],
                          "temperature": 0.95, "max_tokens": 2000},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        last_err = f"http_{resp.status}: {text[:120]}"
                        if resp.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(2 ** attempt); continue
                        return None
                    data = json.loads(text)
                    out = data["choices"][0]["message"]["content"].strip()
                    if out.startswith("```"):
                        out = out.split("```")[1]
                        if out.startswith("json"): out = out[4:]
                    out = out.strip()
                    parsed = json.loads(out)
                    return [{"prompt": (item.get("prompt") or "").strip(),
                             "type": item.get("type", category),
                             "category": category}
                            for item in parsed
                            if isinstance(item, dict) and item.get("prompt")]
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:80]}"
                await asyncio.sleep(2 ** attempt)
        return None

async def main():
    random.seed(42)
    counter = defaultdict(int)
    seen_hash = set()
    if os.path.exists(STAGING):
        with open(STAGING, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    h = hashlib.md5(r["prompt"].encode()).hexdigest()
                    if h not in seen_hash and counter[r["category"]] < TARGET[r["category"]]:
                        seen_hash.add(h)
                        counter[r["category"]] += 1
                except Exception:
                    continue
    print(f"[resume] {dict(counter)}", flush=True)

    fout = open(STAGING, "a", encoding="utf-8")
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(SEM)
        for category in TARGET:
            need = TARGET[category] - counter[category]
            if need <= 0:
                print(f"[{category}] already done", flush=True); continue
            calls_n = int((need + PER_CALL - 1) // PER_CALL * 1.3) + 8
            print(f"\n[{category}] need={need}, planning {calls_n} calls", flush=True)
            call_ids = list(range(calls_n))
            t0 = time.time()
            for wi in range(0, len(call_ids), WAVE):
                wave = call_ids[wi:wi+WAVE]
                tasks = [gen_one(session, sem, cid, category) for cid in wave]
                wave_added = 0
                for fut in asyncio.as_completed(tasks):
                    items = await fut
                    if not items: continue
                    for it in items:
                        p = it["prompt"]
                        if len(p) < 15 or len(p) > 250: continue
                        h = hashlib.md5(p.encode()).hexdigest()
                        if h in seen_hash: continue
                        seen_hash.add(h)
                        if counter[category] >= TARGET[category]: break
                        counter[category] += 1
                        wave_added += 1
                        fout.write(json.dumps(it, ensure_ascii=False) + "\n")
                fout.flush()
                el = time.time() - t0
                print(f"  wave {wi}: +{wave_added}, total={counter[category]}/{TARGET[category]}, el={el:.0f}s", flush=True)
                if counter[category] >= TARGET[category]:
                    print(f"  [{category}] target reached", flush=True); break
            if counter[category] < TARGET[category]:
                print(f"  WARN [{category}] under: {counter[category]}/{TARGET[category]}", flush=True)
    fout.close()

    # 整理 final
    by_cat = defaultdict(list)
    with open(STAGING, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r["category"] in TARGET:
                    by_cat[r["category"]].append(r)
            except Exception:
                continue
    final = []
    for cat in TARGET:
        items = by_cat[cat][:TARGET[cat]]
        for it in items:
            it["domain"] = DOMAIN[cat]
        final.extend(items)
    random.shuffle(final)

    tmp_out = "/tmp/prompts_opus_gen_10k.jsonl"
    with open(tmp_out, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(tmp_out, OUT)

    print(f"\n=== final ===", flush=True)
    cdist = Counter(r["category"] for r in final)
    ddist = Counter(r["domain"] for r in final)
    for c in TARGET:
        print(f"  {c}: {cdist[c]}/{TARGET[c]}")
    print(f"  --- domain ---")
    for d, n in ddist.most_common():
        print(f"  {d}: {n}")
    print(f"\ntotal: {len(final)}, wrote -> {OUT}", flush=True)

asyncio.run(main())
