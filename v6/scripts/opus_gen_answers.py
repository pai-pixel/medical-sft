"""Phase 3: Opus 4.7 造 20k answers with thinking。

改自 v3/dpo/09_taskC_opus_chosen.py 骨架 80%,加:
- category-aware SYS prompts (mcq / acute / tcm / short)
- 强制 <think>...</think> + final answer 输出格式
- staging + resume, SEM=8, WAVE=200, retry 4
- max_tokens=5120 (v5 已验)

输入: merged_prompts_20k.jsonl
输出: answers/opus_all.jsonl.staging (含所有 category)
"""
import os, json, asyncio, resource, time, shutil, re
import aiohttp
from collections import Counter, defaultdict

_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _h), _h))
print(f"[gen] fd: {_s} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}", flush=True)

api_key = None
with open("/mnt/data/huangjiawei/.config/dashscope.env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DASHSCOPE_API_KEY="):
            api_key = line.split("=", 1)[1].strip().strip('"').strip("'"); break
assert api_key and api_key.startswith("ak-")

URL = "https://models-proxy.stepfun-inc.com/v1/chat/completions"
MODEL = os.environ.get("OPUS_MODEL", "claude-opus-4-6:ksyun-aws")  # 默认切金山云 AWS 代理(anthropic 官方渠道不稳)
TEACHER_TAG = os.environ.get("OPUS_TEACHER_TAG", MODEL.replace(":", "-").replace(".", ""))
SEM = int(os.environ.get("OPUS_SEM", 50))  # SEM=50 已实证稳跑, 可覆盖
WAVE = int(os.environ.get("OPUS_WAVE", 200))
MAX_TOKENS = int(os.environ.get("OPUS_MAX_TOKENS", 5120))

V6_DIR = "/mnt/data/huangjiawei/datasets_local/medical_v6"
INP = os.environ.get("OPUS_INP", os.path.join(V6_DIR, "merged_prompts.jsonl"))
STAGING = os.environ.get("OPUS_STAGING", os.path.join(V6_DIR, "answers", "opus_all.jsonl.staging"))
os.makedirs(os.path.dirname(STAGING), exist_ok=True)

# 通用硬规则(继承 v5 SYSTEM_V5)
COMMON_RULES = """核心硬规则:
1. 必须给出明确的临床判断或鉴别诊断方向(信息不足时按现有信息给最可能判断, 再说明需补什么检查)
2. 必须给出具体的方剂/药品、剂量、用法
3. 必须列出禁忌、警示、不适合人群
4. 严禁用"建议立即就医""请咨询医生""无法替代医生面诊""仅供参考"作为开头或主体
5. 危急情况(休克/急性心梗/严重外伤/意识障碍)可建议急救,但先给院前可执行方案
6. **全中文回答, 包括 <think> 推理段也必须全中文**。严禁在 thinking 或 final answer 里使用英文推理/解释/短语(标准计量单位 mg/ml/g 除外; 药名/化合物名可用中文括号附英文, 例如 "异烟肼(INH)"; 但整句话不允许英文)。

输出格式(必须):
<think>
用中文分析问题类型, 抓关键信息, 用中文逻辑推理, 排除干扰选项(如 MCQ), 用中文给出思路。
</think>
<最终中文回答>
"""

SYS_MCQ = f"""你是资深医学教育专家。收到一道医学客观选择题, 请:

{COMMON_RULES}

MCQ 特殊要求:
- 最终回答格式: "答案: X" (X ∈ A/B/C/D), 然后给"解析: <200-500字详细解析每个选项>"
- 解析必须逐项分析 A/B/C/D 为什么对/错
- 涉及中医题要辨证准确, 不生造证型
- 涉及药物要给具体机制/剂量/禁忌
"""

SYS_ACUTE = f"""你是急诊经验丰富的临床医师。收到紧急场景咨询, 请:

{COMMON_RULES}

急症特殊要求:
- 先判断是否需要立即急诊(120/急诊/院前),给红旗症状清单
- 给院前可执行的处理清单(3-5 步, 具体动作)
- 给具体用药(剂量/年龄段/禁忌)
- 儿童/孕妇/老人剂量特别标注
- 焦急口吻要安抚但不能敷衍
"""

SYS_TCM = f"""你是资深中医师, 精通经方时方与辨证施治。收到中医咨询, 请:

{COMMON_RULES}

中医特殊要求:
- 方剂类回答必须包含: 出处 / 组成 / 剂量 / 功用 / 主治 / 方义 / 加减化裁 / 使用注意
- 辨证类回答必须包含: 证型判断 / 治法 / 代表方剂 / 加减 / 中西医对照 / 生活调摄
- 剂量用克(g)标注, 传统"钱/两"括号附
- 严禁生造证型、方名、药名
"""

SYS_SHORT = f"""你是家庭医生朋友式的临床医师。收到极短口语咨询, 请:

{COMMON_RULES}

短问特殊要求:
- 最终回答控制在 300-500 字(比其他类别短)
- 直接切入, 无客套
- 先给最可能的判断和处理, 再说什么情况要就医
- 语气亲和, 但内容专业不敷衍
"""

CATEGORY_SYS = {
    "mcq": SYS_MCQ,
    "mcq_gen": SYS_MCQ,
    "acute": SYS_ACUTE,
    "tcm": SYS_TCM,
    "short": SYS_SHORT,
}


async def gen_one(session, sem, item):
    cat = item.get("category", "short")
    # category 规范化
    if cat == "mcq_gen": cat = "mcq"
    sys_prompt = CATEGORY_SYS.get(cat, SYS_SHORT)
    user_prompt = item["prompt"]

    # MCQ 有 gt_answer 就在 user 段追加一个 hint(帮助 Opus 对齐答案),但只对 CMB 抽的 1k 起作用
    gt = item.get("gt_answer")
    if cat == "mcq" and gt and gt.strip().upper() in "ABCDE":
        user_prompt = user_prompt + f"\n\n(注: 已知标准答案 {gt.strip().upper()}, 请以此为准撰写详细解析。)"

    last_err = "no_exception"
    async with sem:
        for attempt in range(4):
            try:
                # Claude 系(ksyun-aws / anthropic-native)要求 temperature 和 top_p 二选一
                # 网关代理侧会自动加 top_p, 所以我们只用 top_p 不用 temperature
                # top_p: MCQ 收敛(0.7), 其他发散(0.95)
                payload = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "top_p": 0.7 if cat == "mcq" else 0.95,
                    "max_tokens": MAX_TOKENS,
                }
                async with session.post(
                    URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=240),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        last_err = f"http_{resp.status}: {text[:100]}"
                        if resp.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(2 ** attempt); continue
                        return None  # 403 / 400 立即放弃
                    data = json.loads(text)
                    ch = data["choices"][0]
                    answer = ch["message"]["content"].strip()
                    finish = ch.get("finish_reason", "stop")
                    if len(answer) < 50:
                        last_err = f"answer_too_short: {len(answer)}"
                        await asyncio.sleep(2 ** attempt); continue
                    return {
                        "id": item["id"],
                        "prompt": item["prompt"],  # 原 prompt (不含 hint)
                        "category": cat,
                        "source": item.get("source", "unknown"),
                        "gt_answer": gt,
                        "answer": answer,
                        "finish_reason": finish,
                        "teacher": TEACHER_TAG,
                    }
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:150]}"
                await asyncio.sleep(2 ** attempt)
        # 打详细 error 到 log
        print(f"  [FAIL id={item['id']} cat={cat}] {last_err}", flush=True)
        return None


async def main():
    # load prompts
    all_items = []
    with open(INP, encoding="utf-8") as f:
        for line in f:
            all_items.append(json.loads(line))
    print(f"[gen] loaded {len(all_items)} prompts", flush=True)

    # resume
    done = set()
    if os.path.exists(STAGING):
        with open(STAGING, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line); done.add(r["id"])
                except Exception:
                    continue
    print(f"[gen] resume: {len(done)} already", flush=True)
    todo = [x for x in all_items if x["id"] not in done]
    print(f"[gen] todo after resume: {len(todo)}", flush=True)
    cat_dist = Counter(x.get("category", "?") for x in todo)
    print(f"[gen] category dist: {dict(cat_dist)}", flush=True)

    if not todo:
        print("[gen] all done", flush=True); return

    fout = open(STAGING, "a", encoding="utf-8")
    t0 = time.time()
    consecutive_403 = 0
    connector = aiohttp.TCPConnector(limit=SEM * 2, limit_per_host=SEM * 2, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(SEM)
        for wi in range(0, len(todo), WAVE):
            wave = todo[wi:wi+WAVE]
            tasks = [gen_one(session, sem, x) for x in wave]
            ok = 0
            fail = 0
            length_cnt = 0
            for fut in asyncio.as_completed(tasks):
                r = await fut
                if r:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    ok += 1
                    if r.get("finish_reason") == "length":
                        length_cnt += 1
                else:
                    fail += 1
            fout.flush()
            el = time.time() - t0
            done_now = wi + len(wave)
            eta = el / done_now * (len(todo) - done_now) if done_now > 0 else 0
            length_pct = length_cnt / max(ok, 1) * 100
            alert = " ⚠ LEN" if length_pct > 2.0 else ""
            print(f"  wave {wi}/{len(todo)}: +{ok}/{len(wave)} fail={fail} len={length_cnt}({length_pct:.1f}%){alert}, "
                  f"el={el/60:.1f}m ETA={eta/60:.0f}m", flush=True)
            # 早停: 一整个 wave 全 fail = key 挂了
            if fail == len(wave) and len(wave) >= 20:
                consecutive_403 += 1
                if consecutive_403 >= 2:
                    print(f"  ABORT: 2 consecutive full-fail waves, key or model likely dead", flush=True)
                    break
            else:
                consecutive_403 = 0
    fout.close()
    print(f"\n[gen] total {(time.time()-t0)/60:.1f}m", flush=True)

    # summary
    kept = defaultdict(int)
    with open(STAGING, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line); kept[r["category"]] += 1
            except Exception:
                continue
    print(f"[gen] kept by cat: {dict(kept)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
