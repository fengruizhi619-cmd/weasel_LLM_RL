#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下一句预测：用青简那套「提示词 + JSON 契约」跑我们的 Chat 模型，对比 Base 直续写。

问题：我们手上的 Chat 模型能不能**直接**做"下一句预测"，还是必须先微调？
方法：同一批打字留出样本，四个条件对比 ——
    base_greedy   Base 模型贪心继续写（我们现在线上的路子）
    chat_plain    Chat 模型只给上下文，直接问下一句（最短提示）
    chat_json     Chat 模型 + 青简式 JSON 契约（模式 A：只要下一句）
    chat_json_loc Chat 模型 + JSON 契约 + "本地已经能给的候选"（照抄青简的防重复设计）

数据：从加密的打字语料折回连续文本流，取长度 ≥ --min-target 的"下一句"作为目标。
指标：首字命中 / 首3字命中 / 最长公共前缀 / 字符级 F1。

用法：
    python next_sentence_probe.py --model-base <Base路径> --model-chat <Chat路径> --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "tools", "LlamaTreeExp"))
sys.path.insert(0, TREE)

STOP = "。！？；\n"
SENT_END = "。！？…"


# ---------------------------------------------------------------- 数据

def build_streams(records):
    """把碎记录折回连续文本流。ctx 尾接上一段才拼接，否则另起一条流。
    同一段文字被重复打过的，按前 80 字去重（打字语料里重打/重写很常见）。"""
    streams, cur, seen = [], "", set()
    for r in records:
        seg = (r.get("segment") or "").strip()
        ctx = r.get("ctx") or ""
        if not seg or str(r.get("kind", "")).startswith("backspace"):
            continue
        if cur and ctx.endswith(cur[-min(len(cur), 60):]):
            cur += seg
        elif not cur or len(seg) >= len(cur) or cur.endswith(ctx[-min(len(ctx), 10):]):
            cur = (ctx + seg) if not cur else cur + seg
        else:
            if len(cur) >= 40:
                streams.append(cur)
            cur = ctx + seg
        if len(cur) >= 2000:
            streams.append(cur)
            cur = ""
    if len(cur) >= 40:
        streams.append(cur)
    out = []
    for s in streams:
        k = s[:80]
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def make_cases(streams, ctx_len, max_target, min_target, limit, per_stream=3):
    """从流里取 (上下文, 下一句)。跨流轮转取样 —— 否则会全取第一篇笔记，结论不可推广。
    下一个字符若是标点则跳过（那是标点接续，不是句子）。"""
    pools = []
    for si, s in enumerate(streams):
        got, i = [], ctx_len
        while i < len(s) - min_target and len(got) < per_stream:
            nxt = s[i]
            if nxt in SENT_END + "，、）】":
                i += 1
                continue
            j = i
            while j < len(s) and s[j] not in STOP and j - i < max_target:
                j += 1
            tgt = s[i:j].strip()
            if len(tgt) >= min_target:
                got.append({"ctx": s[max(0, i - ctx_len):i], "target": tgt, "stream": si})
            i = j + 1
        if got:
            pools.append(got)
    cases, r = [], 0
    while len(cases) < limit and any(len(p) > r for p in pools):
        for p in pools:
            if len(p) > r and len(cases) < limit:
                cases.append(p[r])
        r += 1
    return cases[:limit]


# ---------------------------------------------------------------- 提示词（照青简的结构）

SYS_JSON = (
    "你是一个拼音输入法的下一句预测引擎。用户会给你一段已经打好的文本（before），"
    "请预测用户接下来最可能打出的**一整句**。\n"
    "要求：\n"
    "- 只预测紧接在 before 后面的内容，不要把 before 的内容抄进答案；\n"
    "- 长度控制在 5 到 30 个字，像一个自然的中文句子，不要解释、不要加引号、不要编号；\n"
    "- 必须接着 before 的语气与话题（技术笔记就接着写技术，叙事就接着叙事）；\n"
    "- 只输出约定的 JSON，其余一概不要。\n"
    '输出 JSON：{"sentence": "…"}'
)

SYS_JSON_LOC = (
    SYS_JSON.replace(
        '输出 JSON：{"sentence": "…"}',
        "你还会收到 local_candidates（本地词库已有的候选，第一个是本地首选）。"
        "它们只是本地猜测，可能全错：和它们重复的不必再给，你的价值在于给出本地给不出、"
        "但接着 before 最自然的**整句**。\n"
        '输出 JSON：{"sentence": "…"}'
    )
)


def user_msg(ctx, local_candidates=None):
    payload = {"before": ctx}
    if local_candidates is not None:
        payload["local_candidates"] = local_candidates
    return json.dumps(payload, ensure_ascii=False)


def strip_think(text):
    """Qwen3 Chat 默认思考模式，会先吐 <think>…</think>。取答案前先剥掉。"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    t = re.sub(r"^.*?</think>", "", t, flags=re.S)   # 只有闭合标签（模板已开思考块）
    t = re.sub(r"<think>.*$", "", t, flags=re.S)     # 思考被 max_new 截断
    return t.strip()


def parse_sentence(text):
    """从回复里取句子：剥思考块 → 先按 JSON 解；被 max_new 截断的残缺 JSON 用正则兜。"""
    t = strip_think(text)
    try:
        obj = json.loads(t[t.find("{"):t.rfind("}") + 1])
        s = obj.get("sentence") or ""
        if isinstance(s, str) and s.strip():
            return s.strip()
    except Exception:
        pass
    m = re.search(r'"sentence"\s*[:：]\s*"(.*?)(?<!\\)"', t, re.S)
    if not m:
        m = re.search(r'"sentence"\s*[:：]\s*"(.*)$', t, re.S)   # 截断，没有收尾引号
    if m:
        s = m.group(1).replace('\\"', '"').replace("\\n", " ").strip()
        if s:
            return s
    t = re.sub(r'^\s*[{"\']?sentence["\']?\s*[:：]\s*', "", t)
    t = t.strip().strip('"\'{}').strip()
    return t.splitlines()[0].strip() if t else ""


# ------------------------------------------------- 语义相似度（本地向量服务，可选）

EMB_URL = os.environ.get("EMB_URL", "http://127.0.0.1:8082/v1/embeddings")


def embed(texts):
    """调本地 llama.cpp 向量服务。调不到就返回 None（不报错，只是没有这一列）。"""
    import urllib.request
    try:
        body = json.dumps({"input": [t if t.strip() else "空" for t in texts],
                           "model": "qwen3-embed"}).encode("utf-8")
        req = urllib.request.Request(EMB_URL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))["data"]
        data = sorted(data, key=lambda d: d.get("index", 0))
        vecs = []
        for d in data:
            v = torch.tensor(d["embedding"], dtype=torch.float32)
            vecs.append(v / (v.norm() + 1e-9))
        return vecs
    except Exception as e:
        print("  [语义列跳过] 向量服务不可用：%s" % e)
        return None


def sem_col(preds, tgts, shuffle_floor=False):
    """预测与目标的平均余弦相似度；shuffle_floor 时错位配对，作为随机下限。"""
    if shuffle_floor:
        tgts = tgts[1:] + tgts[:1]
    vp, vt = embed(preds), embed(tgts)
    if vp is None or vt is None or len(vp) != len(vt):
        return None
    return sum(float(a @ b) for a, b in zip(vp, vt)) / max(1, len(vp))


# ---------------------------------------------------------------- 指标

def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def char_f1(pred, tgt):
    from collections import Counter
    cp, ct = Counter(pred), Counter(tgt)
    hit = sum((cp & ct).values())
    if not pred or not tgt:
        return 0.0
    p = hit / len(pred)
    r = hit / len(tgt)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def score(pred, tgt, ctx=""):
    """echo = 预测开头直接把上下文尾巴抄了一遍（青简提示词专门要防的病）。"""
    tail = ctx[-12:]
    echo = 0.0
    if pred and len(pred) >= 4 and pred[:4] in tail:
        echo = 1.0
    elif pred and len(pred) >= 2 and pred[:2] in tail:
        echo = 0.5
    return {
        "hit1": 1.0 if pred[:1] == tgt[:1] else 0.0,
        "hit3": 1.0 if pred[:3] == tgt[:3] else 0.0,
        "hit_full": 1.0 if pred == tgt else 0.0,
        "lcp": lcp(pred, tgt),
        "f1": char_f1(pred, tgt),
        "echo": echo,
    }


# ---------------------------------------------------------------- 生成

@torch.no_grad()
def gen(model, tok, text, device, max_new=48, greedy=True, temp=0.7, top_p=0.9):
    ids = tok(text, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=not greedy,
                         temperature=temp if not greedy else None,
                         top_p=top_p if not greedy else None,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


def chat_reply(model, tok, system, user, device, max_new=96, thinking=False, shots=None):
    """shots = [(user, assistant), …] 多轮示例，用来逼近青简那种"提示词里带样例"的工程强度。"""
    msgs = [{"role": "system", "content": system}]
    for u, a in (shots or []):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user})
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking)
    except Exception:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return gen(model, tok, text, device, max_new=max_new, greedy=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base", default=r"E:\DSH_data\研究\models\Qwen3-0.6B-Base")
    ap.add_argument("--model-chat", default=r"E:\DSH_data\研究\models\Qwen3-0.6B-Chat")
    ap.add_argument("--segments", default=os.path.join(TREE, "diag", "segments.jsonl"))
    ap.add_argument("--key-file", default=os.path.join(TREE, "diag", "corpus.key"))
    ap.add_argument("--n", type=int, default=40, help="评测样本数")
    ap.add_argument("--ctx-len", type=int, default=120)
    ap.add_argument("--min-target", type=int, default=6)
    ap.add_argument("--max-target", type=int, default=20)
    ap.add_argument("--thinking", action="store_true", help="开启 Chat 思考模式（默认关，青简式调用也不开）")
    ap.add_argument("--show", type=int, default=6, help="明细打印条数")
    ap.add_argument("--out", default=os.path.join(HERE, "next_sentence_result.json"))
    args = ap.parse_args()

    import corpus as cm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = cm.CorpusWriter(args.segments, key_file=args.key_file).read_all()
    streams = build_streams(recs)
    cases = make_cases(streams, args.ctx_len, args.max_target, args.min_target, args.n)
    print("打字记录 %d 条 → 文本流 %d 条 → 评测样本 %d 条（取自 %d 条不同文本流）"
          % (len(recs), len(streams), len(cases), len({c["stream"] for c in cases})))
    for c in cases[:3]:
        print("  ctx=…%r  →  目标=%r" % (c["ctx"][-24:], c["target"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_chat)
    print("加载 Base …"); base = AutoModelForCausalLM.from_pretrained(args.model_base, dtype=torch.float16).to(device).eval()
    print("加载 Chat …"); chat = AutoModelForCausalLM.from_pretrained(args.model_chat, dtype=torch.float16).to(device).eval()

    conds = ["base_greedy", "chat_plain", "chat_json", "chat_json_fs", "chat_json_loc"]
    agg = {c: {"hit1": 0.0, "hit3": 0.0, "hit_full": 0.0, "lcp": 0.0, "f1": 0.0, "echo": 0.0}
           for c in conds}
    details = []
    t0 = time.time()
    n_case = len(cases)
    for k, case in enumerate(cases):
        ctx, tgt = case["ctx"], case["target"]
        out, raws = {}, {}
        # 1) Base 贪心续写
        base_pred = gen(base, tok, ctx, device, max_new=32).strip().split("\n")[0]
        out["base_greedy"] = base_pred
        # 2) Chat 裸问（关思考，否则 max_new 全被 <think> 吃掉）
        raw = chat_reply(chat, tok,
            "你是输入法，接着用户已打的内容写出下一句，只写这一句，不要解释。",
            ctx, device, thinking=args.thinking)
        raws["chat_plain"] = raw
        out["chat_plain"] = strip_think(raw).split("\n")[0].strip()
        # 3) Chat + JSON 契约
        raw = chat_reply(chat, tok, SYS_JSON, user_msg(ctx), device, thinking=args.thinking)
        raws["chat_json"] = raw
        out["chat_json"] = parse_sentence(raw)
        # 4) Chat + JSON 契约 + few-shot（逼近青简那种"提示词里塞样例"的工程强度）
        #    样例取自别的文本流（错开 7 条），不泄漏当前答案。
        shots = []
        for d in (1, 2):
            c2 = cases[(k + 7 * d) % n_case]
            shots.append((user_msg(c2["ctx"]),
                          json.dumps({"sentence": c2["target"]}, ensure_ascii=False)))
        raw = chat_reply(chat, tok, SYS_JSON, user_msg(ctx), device,
                         thinking=args.thinking, shots=shots)
        raws["chat_json_fs"] = raw
        out["chat_json_fs"] = parse_sentence(raw)
        # 5) Chat + JSON 契约 + 本地候选（照青简的防重复设计）
        #    本地候选必须无泄漏：用 Base 模型自己的贪心首二字当"本地首选"，
        #    绝不能用 tgt 的任何字符（否则等于把答案喂进去）。
        loc = [x for x in dict.fromkeys([base_pred[:2], base_pred[2:4]]) if x] or ["（无）"]
        raw = chat_reply(chat, tok, SYS_JSON_LOC, user_msg(ctx, loc), device, thinking=args.thinking)
        raws["chat_json_loc"] = raw
        out["chat_json_loc"] = parse_sentence(raw)

        for c in conds:
            s = score(out[c], tgt, ctx)
            for kk in agg[c]:
                agg[c][kk] += s[kk] / len(cases)
        details.append({"ctx": ctx, "target": tgt, "local_cand": loc, **out,
                        "raw": {k: v[:300] for k, v in raws.items()}})
        if (k + 1) % 5 == 0:
            print("  %d/%d  %.0fs" % (k + 1, len(cases), time.time() - t0), flush=True)

    print("\n=== 汇总（%d 条，%.0f 秒）===" % (len(cases), time.time() - t0))
    print("%-16s %6s %6s %6s %7s %6s %6s" % ("条件", "首字", "首3字", "整句", "LCP均", "F1", "抄上下文"))
    for c in conds:
        a = agg[c]
        print("%-16s %6.3f %6.3f %6.3f %7.2f %6.3f %6.3f"
              % (c, a["hit1"], a["hit3"], a["hit_full"], a["lcp"], a["f1"], a["echo"]))

    # 语义对照：字面命中 Base 赢，但要看清 Chat 是不是"语义对、字面不同"
    print("\n=== 语义相似度（本地 Qwen3-Embedding，余弦）===")
    tgts = [d["target"] for d in details]
    tops = sem_col([d["ctx"][-30:] for d in details], list(tgts))
    chanc = sem_col([d["ctx"][-30:] for d in details], list(tgts), shuffle_floor=True)
    if chanc is not None:
        print("  参考：上下文尾 30 字 ↔ 目标（只靠话题相邻就能拿到的分）%.3f ｜ 错位配对（随机下限）%.3f"
              % (tops, chanc))
        for c in conds:
            s = sem_col([d[c] or "空" for d in details], list(tgts))
            print("  %-16s %.3f" % (c, s if s is not None else float("nan")))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n": len(cases), "agg": agg, "details": details}, f, ensure_ascii=False, indent=2)
    print("\n前 %d 条明细：" % args.show)
    for d in details[:args.show]:
        print("  目标: %s   [本地候选 %s]" % (d["target"], d.get("local_cand")))
        for c in conds:
            print("    %-15s %s" % (c, d[c][:44]))
        print("    chat_raw        %s" % d["raw"]["chat_json"][:100].replace("\n", "⏎"))
    print("\n已写 %s" % args.out)


if __name__ == "__main__":
    main()
