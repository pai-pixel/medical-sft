"""A_03: Opus 4.7 造补 seed prompts (10k+)。

改自 v3/dpo/06_full_opus_prompt_gen.py 骨架。

4 categories:
- mcq_gen: 4000 — 造 MCQ (4 选项 A/B/C/D + 自标答案 + explanation)
- acute:   2000 — 急症场景 (儿科/孕妇/急救/中毒)
- tcm:     4000 — 中医方剂/辨证问答
- short:   1000 — 100 字内真实口语提问

网关: claude-opus-4-7:mindracode-anthropic-qianli (probe 2.62s ✅ 2026-07-01)
SEM=6, WAVE=100, retry 4, hash 去重
预算: ~10k prompts / 10 per call = ~1000 calls × 3s = ~50 min 纯网络 + 抖动 60-90 min
"""
import os, json, asyncio, resource, random, hashlib, time, shutil, re
import aiohttp
from collections import defaultdict, Counter

_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _h), _h))
print(f"[A_03] fd: {_s} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}", flush=True)

api_key = None
with open("/mnt/data/huangjiawei/.config/dashscope.env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DASHSCOPE_API_KEY="):
            api_key = line.split("=", 1)[1].strip().strip('"').strip("'"); break
assert api_key and api_key.startswith("ak-")

URL = "https://models-proxy.stepfun-inc.com/v1/chat/completions"
MODEL = "claude-opus-4-7:mindracode-anthropic-qianli"
PER_CALL = 10
SEM = 50  # 激进试, 若 429 触发多回退 SEM=8, retry 4 次自动退避
WAVE = 100  # SEM 大了 wave 也可以放大, 每 wave ~3 min

OUT_DIR = "/mnt/data/huangjiawei/datasets_local/medical_v6/seed_prompts"
STAGING = os.path.join(OUT_DIR, "opus_gen_seeds.jsonl.staging")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = {
    "mcq_gen":  4000,
    "acute":    2000,
    "tcm":      4000,
    "short":    1000,
}

TOPICS = {
    "mcq_gen": [
        "内科(呼吸/心血管/消化/内分泌)常见病诊断鉴别",
        "外科急腹症/创伤/普外常见诊断决策",
        "妇产科(月经/孕产/产后/更年期)诊治",
        "儿科(发育/常见病/儿童用药)诊疗",
        "神经科(头痛/癫痫/脑卒中/周围神经)诊断",
        "皮肤科常见病症鉴别与用药",
        "药理与药物相互作用(抗生素/降压/降糖/抗凝)",
        "中医基础(阴阳五行/脏腑经络/四诊八纲)",
        "中医方剂(经方时方组成、君臣佐使、配伍加减)",
        "中药学(性味归经、毒性、配伍禁忌、十八反十九畏)",
        "医学基础(生化/病理/病理生理/免疫)客观题",
        "公共卫生与预防医学(流行病学/传染病/职业病)",
    ],
    "acute": [
        "婴幼儿(0-3 岁)发热/惊厥/腹泻/呼吸急促",
        "孕妇孕期常见急症(先兆流产/妊高症/胎动异常)",
        "老人跌倒/骨折/意识改变/胸痛",
        "急性中毒(药物过量/农药/煤气/食物)家庭急救",
        "外伤/大出血/烧烫伤/触电/溺水现场处理",
        "急性过敏反应/哮喘发作/低血糖/低血压",
        "急性胸痛/心悸/呼吸困难 家人如何辨别",
        "儿童误吞异物/药物/危险物品",
        "中暑/热射病/严寒暴露 户外场景",
        "夜间突发不适 是否要立即去急诊 的判断",
    ],
    "tcm": [
        "常见证型辨证(阴虚/阳虚/气虚/血虚/痰湿/湿热) 问病求治",
        "经方应用(桂枝汤/小柴胡汤/四逆汤/白虎汤类) 咨询",
        "时方与常用方(四君子/六味地黄/逍遥散/补中益气) 使用",
        "妇科中医(月经不调/更年期/产后/带下) 辨证",
        "儿科中医(积滞/惊风/夜啼/遗尿) 辨证",
        "内科杂病(失眠/眩晕/心悸/胃痛/腹泻) 中医治疗",
        "中药配伍与煎服(先煎/后下/包煎/久煎)常识",
        "中西医结合治疗(高血压/糖尿病/肿瘤/慢病)咨询",
        "常见食疗与养生(春夏秋冬/体质调理)",
        "针灸推拿常见适应症与禁忌",
        "膏方 / 药酒 / 药膳 家用咨询",
        "疑难杂症与中医外治(艾灸/拔罐/刮痧)",
    ],
    "short": [
        "身体某处不舒服 10-30 字口语描述(带情绪)",
        "药能不能一起吃 30-50 字快速咨询",
        "老人小孩临时不适 家人 40-70 字焦虑问",
        "查体报告某个指标偏高 20-40 字简短问",
        "常见小症状(头晕/嗓子疼/胃胀/皮疹)口语问",
        "药物剂量能不能加/减 短问",
        "换季/劳累/失眠后 短问",
        "月经异常 短问",
        "刚吃过某种食物/药物有反应 短问",
        "复诊药到期该不该继续 短问",
    ],
}

DESC = {
    "mcq_gen": "医学客观选择题, 4 选项 A/B/C/D, 单选. 覆盖内外妇儿+中医+基础. 难度中偏难",
    "acute":   "紧急场景, 家人焦急口吻求助, 常带 '怎么办' '要不要送急诊' 之类",
    "tcm":     "中医辨证 / 方剂 / 中药相关咨询. 真实口吻可以中医术语混口语",
    "short":   "极短口语 10-100 字, 真用户风格, 可不完整, 带情绪 / 焦虑 / 不耐烦",
}

# 每类的 output schema 描述
CLASS_SCHEMA = {
    "mcq_gen": (
        '每项 {{"prompt": "<题干>\\nA. ...\\nB. ...\\nC. ...\\nD. ...", '
        '"gt_answer": "<A|B|C|D>", '
        '"topic": "<话题>", '
        '"category": "mcq_gen"}}'
    ),
    "acute": '每项 {{"prompt": "...", "topic": "<话题>", "category": "acute"}}',
    "tcm":   '每项 {{"prompt": "...", "topic": "<话题>", "category": "tcm"}}',
    "short": '每项 {{"prompt": "...", "topic": "<话题>", "category": "short"}}',
}

META_MCQ = """你是资深医学教育出题老师。请生成 {n} 道原创中文医学客观选择题(单选,4 选项 A/B/C/D),覆盖主题:

{seeds}

要求:
1. 全中文,不用英文药名缩写(可括号附)
2. 题干具体、场景清晰,不要"下列哪项是XX"这种教科书式空泛问法
3. 4 个选项都有干扰性(不要"以上都对"这种偷懒选项)
4. 标准答案唯一确定
5. 难度中偏难(适合执业医师/规培/本科高年级)
6. 中医题必须辨证准确,不要生造证型

输出严格 JSON list, {n} 项, {schema}. 直接 [ 开始, 无 markdown 标记, 无前后解释。"""

META_OPEN = """你是医疗对话数据集设计师。请生成 {n} 个真实用户向 AI 医疗助手提出的问题, 覆盖主题:

{seeds}

要求:
1. 真实性: 真用户口吻(口语化, 可不完整, 带情绪/焦虑/不耐烦)
2. 多样性: 每条独立, 句式句长变化, 不要模板复用
3. 长度: {len_hint}
4. 内容范围: {desc}
5. 全中文, 不掺英文(药名可括号)

输出严格 JSON list, {n} 项, {schema}. 直接 [ 开始, 无 markdown 标记, 无前后解释。"""

LEN_HINT = {
    "mcq_gen": "题干 40-200 字, 4 选项各 5-40 字",
    "acute":   "50-200 字, 真实焦急场景",
    "tcm":     "40-180 字, 可带证候描述",
    "short":   "10-100 字, 极短口语",
}


def build_meta(category, seeds_str, n):
    if category == "mcq_gen":
        return META_MCQ.format(n=n, seeds=seeds_str, schema=CLASS_SCHEMA[category])
    return META_OPEN.format(
        n=n, seeds=seeds_str, desc=DESC[category],
        len_hint=LEN_HINT[category], schema=CLASS_SCHEMA[category],
    )


async def gen_one(session, sem, call_idx, category):
    seeds = random.sample(TOPICS[category], min(3, len(TOPICS[category])))
    seeds_str = "\n".join(f"- {s}" for s in seeds)
    body = build_meta(category, seeds_str, PER_CALL)

    payload_temp = 0.3 if category == "mcq_gen" else 0.95
    # max_tokens 分类别: 每 call 10 条 JSON, MCQ 单条 ~250-400 字(题干+4 选项), 10 条需 3000-4000 tokens
    # 其他类别单条 100-200 字, 10 条需 1500-2500 tokens
    # 留 1.3x buffer + 向上 256 取整
    max_tok = {
        "mcq_gen": 4096,  # MCQ 最宽裕
        "acute":   3072,  # 急症场景描述长
        "tcm":     3072,  # 中医术语多
        "short":   2048,  # 短问, 但 10 条也要 1500+
    }[category]

    last_err = "no_exception"
    async with sem:
        for attempt in range(4):
            try:
                async with session.post(
                    URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": MODEL,
                          "messages": [{"role": "user", "content": body}],
                          "temperature": payload_temp,
                          "max_tokens": max_tok},
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        last_err = f"http_{resp.status}: {text[:120]}"
                        if resp.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(2 ** attempt); continue
                        # 403 / parameter / not_supported → 立即 return
                        return None
                    data = json.loads(text)
                    ch = data["choices"][0]
                    out = ch["message"]["content"].strip()
                    finish = ch.get("finish_reason", "stop")
                    # 撞顶 → 直接判失败, 不写盘(防截断 JSON 污染)
                    if finish == "length":
                        last_err = f"finish=length (max_tokens={max_tok} 不够)"
                        return {"__truncated__": True, "category": category}
                    # 剥 markdown
                    if out.startswith("```"):
                        out = out.split("```")[1]
                        if out.startswith("json"): out = out[4:]
                    out = out.strip()
                    parsed = json.loads(out)
                    return [{"prompt": (item.get("prompt") or "").strip(),
                             "topic": item.get("topic", ""),
                             "category": item.get("category", category),
                             "gt_answer": item.get("gt_answer") if category == "mcq_gen" else None}
                            for item in parsed
                            if isinstance(item, dict) and item.get("prompt")]
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                last_err = f"{type(e).__name__}: {str(e)[:80]}"
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:80]}"
                await asyncio.sleep(2 ** attempt)
        return None


LEN_RANGE = {
    "mcq_gen": (60, 800),
    "acute":   (30, 260),
    "tcm":     (20, 240),
    "short":   (8, 130),
}


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
                    if h not in seen_hash and counter[r["category"]] < TARGET.get(r["category"], 0):
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
            # 每 call PER_CALL 条, 留 1.5x buffer(重复+过滤损失)
            calls_n = int((need + PER_CALL - 1) // PER_CALL * 1.5) + 10
            lo, hi = LEN_RANGE[category]
            print(f"\n[{category}] need={need}, planning {calls_n} calls, len={lo}-{hi}", flush=True)
            call_ids = list(range(calls_n))
            t0 = time.time()
            for wi in range(0, len(call_ids), WAVE):
                wave = call_ids[wi:wi+WAVE]
                tasks = [gen_one(session, sem, cid, category) for cid in wave]
                wave_added = 0
                dropped_len = 0
                dropped_dup = 0
                truncated_calls = 0
                done_in_wave = 0
                for fut in asyncio.as_completed(tasks):
                    result = await fut
                    done_in_wave += 1
                    if not result:
                        continue
                    # 撞顶信号(dict 单标记)
                    if isinstance(result, dict) and result.get("__truncated__"):
                        truncated_calls += 1
                        continue
                    items = result if isinstance(result, list) else []
                    if not items:
                        continue
                    for it in items:
                        p = it["prompt"]
                        if len(p) < lo or len(p) > hi:
                            dropped_len += 1; continue
                        h = hashlib.md5(p.encode()).hexdigest()
                        if h in seen_hash:
                            dropped_dup += 1; continue
                        seen_hash.add(h)
                        if counter[category] >= TARGET[category]: break
                        counter[category] += 1
                        wave_added += 1
                        fout.write(json.dumps(it, ensure_ascii=False) + "\n")
                    if done_in_wave % 5 == 0:
                        fout.flush()
                        el = time.time() - t0
                        pct_trunc = truncated_calls / max(done_in_wave, 1) * 100
                        alert = " ⚠ TRUNC" if pct_trunc > 2.0 else ""
                        print(f"  [{category}] +{done_in_wave}/{len(wave)} calls, "
                              f"cat={counter[category]}/{TARGET[category]}, "
                              f"trunc={truncated_calls}({pct_trunc:.1f}%){alert}, el={el:.0f}s", flush=True)
                fout.flush()
                el = time.time() - t0
                print(f"  wave {wi}: +{wave_added}, total={counter[category]}/{TARGET[category]}, "
                      f"drop_len={dropped_len} drop_dup={dropped_dup}, el={el:.0f}s", flush=True)
                if counter[category] >= TARGET[category]:
                    print(f"  [{category}] target reached", flush=True); break
            if counter[category] < TARGET[category]:
                print(f"  WARN [{category}] under: {counter[category]}/{TARGET[category]}", flush=True)
    fout.close()

    # 整理 final: 每类分别输出
    by_cat = defaultdict(list)
    with open(STAGING, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r["category"] in TARGET:
                    by_cat[r["category"]].append(r)
            except Exception:
                continue

    for cat in TARGET:
        items = by_cat[cat][:TARGET[cat]]
        # 补 id + source
        for i, it in enumerate(items):
            it["id"] = f"opus_{cat}_{i:05d}"
            it["source"] = f"opus-4-7-generated-{cat}"
            it.setdefault("meta", {"topic": it.get("topic", "")})
            if "topic" in it: it.pop("topic", None)
        out_path = os.path.join(OUT_DIR, f"opus_{cat}_gen.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{cat}] wrote {len(items)} → {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
