#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混料训练：小说文本 + 打字数据，按固定比例逐步交替取样，结束后落盘。

标准设置（来自 experiments/mix_ratio.py 的对照结论）：
    打字 : 小说 = 25 : 75        --ratio 0.25
    lr = 1e-5                    --lr
打字池有两个来源，合并后按 (ctx, 文本) 精确去重：
    diag/corpus.jsonl    在线路径的 accept 记录
    diag/segments.jsonl  离线采集器的 commit 记录
每条记录按 token 逐位展开（标点跳过），所以一条 3 字的记录贡献 3 个训练步。

评估用最后 --holdout 条 accept 记录，它们不进入训练池。

主干冻结：两边的隐状态各算一次，训练只动 lm_head；收尾按五槽稀释方案落盘。
"""
import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import corpus as corpus_mod
import probe_eval as pe
import unified_pipeline as up
import unified_watcher as uw


def walk_steps(engine, text, chunk=400, prompt=64):
    """把一段文本展开成 (隐状态, 真值 token) 列表，标点跳过。"""
    tok, dev = engine.tokenizer, engine.device
    hs, targets = [], []
    pos = 0
    while pos < len(text):
        head = text[max(0, pos - prompt):pos]
        if not head:
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
        for row in range(n_head, len(ids)):
            tid = ids[row]
            if up.is_no_target(tok.decode([tid])):
                continue
            hs.append(h[row - 1].float())
            targets.append(tid)
    return hs, targets


def steps_for(engine, ctx, txt):
    """一段上下文 + 一段用户实际打的文本 -> 只针对 txt 的 (隐状态, 真值 token)。"""
    tok, dev = engine.tokenizer, engine.device
    ids = tok.encode(ctx + txt, add_special_tokens=False)
    n_ctx = len(tok.encode(ctx, add_special_tokens=False)) if ctx else 0
    if len(ids) <= n_ctx:
        return []
    with torch.no_grad():
        h = engine.model.model(
            input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0]
    out = []
    for row in range(max(n_ctx, 1), len(ids)):
        tid = ids[row]
        if up.is_no_target(tok.decode([tid])):
            continue
        out.append((h[row - 1].float(), tid))
    return out


def load_typing_sources():
    """返回 (在线 accept 列表, 离线 commit 列表)，元素为 (ctx, 文本)。"""
    online, offline = [], []
    for r in corpus_mod.CorpusWriter(os.path.join(HERE, "diag", "corpus.jsonl")).read_all():
        if r.get("kind") == "accept" and r.get("ctx") and r.get("typed"):
            online.append((r["ctx"], r["typed"]))
    for r in corpus_mod.CorpusWriter(os.path.join(HERE, "diag", "segments.jsonl")).read_all():
        if r.get("kind") == "commit" and r.get("ctx") and r.get("segment"):
            offline.append((r["ctx"], r["segment"]))
    return online, offline


def build_typing_pool(engine, holdout):
    """训练池 =（在线 accept 去掉评估留出）+ 离线 commit，按 (ctx, 文本) 去重。"""
    online, offline = load_typing_sources()
    hold = online[-holdout:] if holdout else []
    train = (online[:-holdout] if holdout else online) + offline
    seen, uniq = set(), []
    for ctx, txt in train:
        if (ctx, txt) in seen:
            continue
        seen.add((ctx, txt))
        uniq.append((ctx, txt))
    hs, tg = [], []
    for ctx, txt in uniq:
        for h, t in steps_for(engine, ctx, txt):
            hs.append(h)
            tg.append(t)
    return uniq, hold, hs, tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("novel")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--ratio", type=float, default=0.25,
                    help="打字数据占的步数比例（标准 0.25）")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "diag", "checkpoints_mix"))
    ap.add_argument("--init", default=os.path.join(HERE, "diag", "checkpoints_online"))
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--save-every", type=int, default=4000)
    args = ap.parse_args()

    raw = pe.read_text(args.novel)
    novel_text = raw[args.start:args.end or len(raw)]

    engine = up.TreeEngine(up.MODEL_PATH, lr=args.lr, device=up.DEVICE, dtype="float16")
    ckpt = uw.CheckpointManager(engine, args.ckpt_dir)
    if not ckpt.load_latest():
        engine.load_checkpoint(os.path.join(args.init, "lm_head_t0.pt"))
    engine.optimizer.param_groups[0]["lr"] = args.lr
    W = engine.model.lm_head.weight

    print("[mix] 小说 %d 字，展开中 …" % len(novel_text), flush=True)
    t0 = time.time()
    hn, tn = walk_steps(engine, novel_text)
    n_novel = len(tn)
    print("[mix] 小说 %d 步（%.0f 秒）" % (n_novel, time.time() - t0), flush=True)

    t0 = time.time()
    uniq, hold, ht, tt = build_typing_pool(engine, args.holdout)
    n_typ = len(tt)
    print("[mix] 打字池 %d 条去重记录 -> %d 步，评估留出 %d 条（%.0f 秒）"
          % (len(uniq), n_typ, len(hold), time.time() - t0), flush=True)

    total = int(round(n_novel / (1.0 - args.ratio))) if args.ratio < 1 else n_novel
    if args.max_steps:
        total = min(total, args.max_steps)
    n_typ_used = int(round(total * args.ratio))
    print("[mix] 总步数 %d = 小说 %d + 打字 %d（打字池复读 %.1f 遍），lr=%g"
          % (total, total - n_typ_used, n_typ_used,
             n_typ_used / max(1, n_typ), args.lr), flush=True)

    hN = torch.stack(hn); tN = torch.tensor(tn, device=engine.device)
    hT = torch.stack(ht); tT = torch.tensor(tt, device=engine.device)

    rng = random.Random(args.seed)
    order = list(range(n_typ))
    rng.shuffle(order)
    ti = ni = 0
    loss_sum = 0.0
    engine.model.train()
    for step in range(1, total + 1):
        if n_typ and rng.random() < args.ratio:
            k = order[ti % n_typ]; ti += 1
            h, t = hT[k], tT[k]
        else:
            k = ni % n_novel; ni += 1
            h, t = hN[k], tN[k]
        logits = engine.model.lm_head(h.unsqueeze(0))[0]
        loss = -torch.log_softmax(logits, dim=-1)[t]
        engine.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        engine.optimizer.step()
        loss_sum += float(loss.detach())
        if step % 2000 == 0 or step == total:
            print("[mix] %d/%d  平均 loss %.4f  (小说 %d / 打字 %d)"
                  % (step, total, loss_sum / step, ni, ti), flush=True)
        if step % args.save_every == 0:
            engine.model.eval()
            ckpt.mark_dirty()
            ckpt.save_epoch(force=True, slots=("lm_head_t0.pt",))
            engine.model.train()
    engine.model.eval()
    engine.optimizer.zero_grad(set_to_none=True)
    ckpt.mark_dirty()
    ckpt.save_epoch(force=True, slots=("lm_head_t0.pt",))
    print("[mix] 落盘完成 updates=%d -> %s" % (ckpt.updates, args.ckpt_dir), flush=True)
    print("[mix] 权重已写入 %s/lm_head_t0.pt" % args.ckpt_dir, flush=True)


if __name__ == "__main__":
    main()
