#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score one head on held-out text and on the user's own accepted predictions.

usage:
  python probe_eval.py [--text FILE --text-start N --text-n N]
                       [--typing N] [--ckpt PATH] [--tag NAME]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
import corpus
import unified_pipeline as up


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    raise SystemExit("cannot decode " + path)


def text_probe(path, start, n, context=100, step=11):
    text = read_text(path)[start:]
    out = []
    i = context
    while len(out) < n and i < len(text):
        out.append((text[i - context:i], text[i]))
        i += step
    return out


def typing_probe(n, before=None):
    p = os.path.join(HERE, "diag", "corpus.jsonl")
    recs = corpus.CorpusWriter(p).read_all()
    acc = [r for r in recs
           if r.get("kind") == "accept" and r.get("ctx") and r.get("typed")]
    if before:
        acc = [r for r in acc if float(r.get("time", 0)) < before]
    return [(r["ctx"], r["typed"][0]) for r in acc[-n:]]


def score(engine, data):
    top1 = top5 = top20 = 0
    mrr = 0.0
    ranks = []
    for ctx, ch in data:
        ids = engine.tokenizer.encode(ctx, return_tensors="pt").to(engine.device)
        with torch.no_grad():
            logits, _ = engine.backbone_logits(ids)
            probs = F.softmax(logits[0].float(), dim=-1)
            tid = engine.tokenizer.encode(ch, add_special_tokens=False)[0]
            rank = int((probs > probs[tid]).sum().item())
        ranks.append(rank)
        top1 += rank == 0
        top5 += rank < 5
        top20 += rank < 20
        mrr += 1.0 / (rank + 1)
    n = max(1, len(data))
    return (top1 / n * 100, top5 / n * 100, top20 / n * 100,
            mrr / n, sum(ranks) / n)


WALK_HELP = """The three numbers this project steers by, walked one token at a
time over a text (frozen backbone, teacher forcing, one forward per chunk):

  前20累计概率  mean of sum(top-20 probabilities). How much of the mass the
               candidate list actually holds; if the twenty were 0.05 each it
               would be 1.0.
  命中率        the share of scored positions whose next token IS in the top-20.
  脱靶率        1 - 命中率.
  最长链        the longest run of consecutive hits - how many tokens in a row
               the candidate list covered.

Punctuation is skipped on both sides (the tree stops at it, so it is never
asked for) and is transparent inside a chain.
"""


def walk_metrics(engine, text, chunk=400, prompt=64, k=20):
    """Walk `text` and return the three metrics above (None if nothing scored)."""
    tok, dev = engine.tokenizer, engine.device
    W = engine.model.lm_head.weight.detach()
    mass, hit20, hit1, hit5 = [], [], [], []
    scored = 0
    chain = best = 0
    pos = 0
    while pos < len(text):
        head = text[max(0, pos - prompt):pos]
        if not head:
            # the very first character has nothing in front of it: it can only
            # be context, so start one character later instead of dropping the
            # whole first chunk
            pos += 1
            continue
        seg = text[pos:pos + chunk]
        ids = tok.encode(head + seg, add_special_tokens=False)
        n_head = len(tok.encode(head, add_special_tokens=False))
        pos += len(seg)
        if len(ids) <= n_head:
            continue
        with torch.no_grad():
            h = engine.model.model(
                input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0]
            logits = h[n_head - 1:-1].to(W.dtype) @ W.T
            probs = torch.softmax(logits.float(), dim=-1)
            top = probs.topk(min(k, probs.shape[-1]), dim=-1)
        for row in range(logits.shape[0]):
            target = ids[n_head + row]
            if up.is_no_target(tok.decode([target])):
                continue
            scored += 1
            mass.append(float(top.values[row].sum()))
            in20 = bool((top.indices[row] == target).any())
            hit20.append(in20)
            hit1.append(int(top.indices[row][0]) == target)
            hit5.append(bool((top.indices[row][:5] == target).any()))
            chain = chain + 1 if in20 else 0
            if chain > best:
                best = chain
    if not scored:
        return None
    hits = sum(1 for v in hit20 if v)
    return {"scored": scored, "mass20": sum(mass) / scored,
            "hit_rate": 100.0 * hits / scored,
            "miss_rate": 100.0 * (1 - hits / scored),
            "hit1": 100.0 * sum(1 for v in hit1 if v) / scored,
            "hit5": 100.0 * sum(1 for v in hit5 if v) / scored, "longest": best}


def report_walk(r, tag, name):
    if not r:
        print("FLAG WALK %-14s %-16s (没有可打分的位置)" % (tag, name), flush=True)
        return
    print("FLAG WALK %-14s %-16s 打分=%4d  前20累计概率=%.4f  命中率=%.1f%%  "
          "脱靶率=%.1f%%  最长链=%d  (top1 %.1f%% / top5 %.1f%%)"
          % (tag, name, r["scored"], r["mass20"], r["hit_rate"], r["miss_rate"],
             r["longest"], r["hit1"], r["hit5"]), flush=True)


def report(tag, name, data, engine):
    r = score(engine, data)
    print("FLAG RESULT %-20s %-10s(%3d条) top1=%.1f%% top5=%.1f%% top20=%.1f%% MRR=%.3f 平均排名=%.1f"
          % ((tag, name, len(data)) + r), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="")
    ap.add_argument("--text-start", type=int, default=0)
    ap.add_argument("--text-n", type=int, default=200)
    ap.add_argument("--typing", type=int, default=0)
    ap.add_argument("--typing-before", type=float, default=None)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--walk", default="",
                    help="text file to walk for the top-20 mass / hit rate / "
                         "longest chain metrics")
    ap.add_argument("--walk-start", type=int, default=0)
    ap.add_argument("--walk-n", type=int, default=1000)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--model", default=up.MODEL_PATH)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    engine = up.TreeEngine(args.model, lr=up.LR, device=up.DEVICE, dtype=args.dtype)
    if args.ckpt:
        if not engine.load_checkpoint(args.ckpt):
            raise SystemExit("checkpoint load failed: " + args.ckpt)
    print("FLAG >>> %s" % args.tag, flush=True)
    # [TRAIN-035] Punctuation is never a prediction target - the tree stops
    # growing on it - so it must not sit inside the headline number either. The
    # head is right about 。 and ， 70% of the time and about the next word 8% of
    # the time, and a single averaged figure hides which of the two moved.
    if args.text:
        data = text_probe(args.text, args.text_start, args.text_n)
        split(data, args.tag, "留出文本", engine)
    if args.typing:
        data = typing_probe(args.typing, args.typing_before)
        split(data, args.tag, "打字记录", engine)
    if args.walk:
        text = read_text(args.walk)[args.walk_start:args.walk_start + args.walk_n]
        report_walk(walk_metrics(engine, text), args.tag, "全文游走")
    return 0


def split(data, tag, name, engine):
    words = [d for d in data if not up.is_punct(d[1])]
    punct = [d for d in data if up.is_punct(d[1])]
    report(tag, name, data, engine)
    report(tag, name + "-词", words, engine)
    report(tag, name + "-标点", punct, engine)


if __name__ == "__main__":
    sys.exit(main())
