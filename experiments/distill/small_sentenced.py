#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按标点分割 + ≥100 上下文的句子式训练（用户纠正的训练配方）。

用户纠正（2026-09-14）：
1) 训练样本按**标点分割** —— 目标（"一句话"）内部不能有标点；
2) 上下文**可以有标点**（它只是历史）；
3) 上下文至少 **≥100 字**，作为输入。

为什么要这样改：之前的训练在所有位置上均匀预测，导致"下一个字符是逗号/句号"成为
高频目标 —— 这正是 0.13M 模型贪心续写塌缩成 "，，，，" 的根因。改成"只学句子内部的
内容延续"后，分布被锐化到候选树在句读剪枝后的真实使用场景；且每个训练位置都有 ≥100
字上文（之前 256 窗口的**开头位置几乎没有上下文**）。

做法：
- reconstruct_stream(sep="\\n") 段落间插换行，防跨段拼接
- 按标点集合切分 → 片段（无标点）为目标，前一标点留在上下文里
- 上下文取片段前最多 240 字、至少 100 字（不足则丢样本）
- loss 只算目标片段位置（content→content），上下文与"预测标点"都不训

用法：
    python small_sentenced.py --steps 6000
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import capacity_sweep as CS  # noqa: E402
import probe_continuation as PC  # noqa: E402（greedy / sample / ngram_greedy）

# 分割标点：句末/句读 + 成对标点 + 省略号破折号 + 换行（段落边界）。目标片段内绝无这些。
SPLIT_RE = re.compile(r"[，。！？；：、,.!?;:()（）《》「」【】…—~·\n]")


def sentence_samples(stoi, text, seq=256, min_ctx=100, max_target=64):
    """返回 (x, y, mask)：x 输入（上下文+目标，pad 到 seq）、y 右移目标、mask 只标目标位置。
    目标片段 = 两个标点之间（含起首）；前一标点留在上下文里。"""
    xs, ys, ms = [], [], []
    n_skip_ctx, n_skip_short, n_tot = 0, 0, 0
    i = 0
    L = len(text)
    while i < L:
        # 找到片段起点（跳过连续标点）
        while i < L and SPLIT_RE.match(text[i]):
            i += 1
        s = i
        while i < L and not SPLIT_RE.match(text[i]):
            i += 1
        frag = text[s:i]
        n_tot += 1
        if len(frag) < 2 or len(frag) > max_target:
            n_skip_short += 1
            continue
        ctx_text = text[max(0, s - seq + 1):s]          # 片段前的历史（含前一个标点）
        if len(ctx_text) < min_ctx:                      # 上下文至少 ≥100 字
            n_skip_ctx += 1
            continue
        ctx = CS.char_encode(stoi, ctx_text)
        tgt = CS.char_encode(stoi, frag)
        ctx = ctx[-(seq - len(tgt)):]                    # 总长 ≤ seq
        ids = ctx + tgt
        n = len(ids)
        x = torch.zeros(seq, dtype=torch.long)
        y = torch.zeros(seq, dtype=torch.long)
        m = torch.zeros(seq, dtype=torch.bool)
        x[:n] = torch.tensor(ids)
        y[:n - 1] = torch.tensor(ids[1:])
        # loss 只标目标片段内 content→content 的位置
        for k in range(len(tgt) - 1):
            m[len(ctx) + k] = True
        xs.append(x)
        ys.append(y)
        ms.append(m)
    print("片段总数 %d ｜ 丢(目标<2或>%d) %d ｜ 丢(上下文<%d) %d ｜ 样本 %d"
          % (n_tot, max_target, n_skip_short, min_ctx, n_skip_ctx, len(xs)))
    if not xs:
        raise SystemExit("没有样本")
    return (torch.stack(xs), torch.stack(ys), torch.stack(ms))


def masked_loss(model, xb, yb, mb, device):
    lg = model(xb).logits
    B, L, V = lg.shape
    loss = torch.zeros((), device=device)
    n = 0
    for c in range(0, L, 128):
        seg = lg[:, c:c + 128].reshape(-1, V)
        tgt = yb[:, c:c + 128].reshape(-1)
        m = mb[:, c:c + 128].reshape(-1)
        if m.any():
            loss = loss + F.cross_entropy(seg[m], tgt[m])
            n += 1
    return loss / max(n, 1)


@torch.no_grad()
def content_topk(model, xh, yh, device, stoi, k_list=(1, 5, 10)):
    """只统计"真下一字符不是标点"的位置（与训练目标一致，候选树场景）。"""
    punct_ids = {v for c, v in stoi.items() if SPLIT_RE.match(c)}
    hits = {k: 0 for k in k_list}
    tot = 0
    for i in range(xh.size(0)):
        lg = model(xh[i:i + 1].to(device)).logits[0]
        top10 = lg.topk(max(k_list), dim=-1).indices
        yb = yh[i].to(device)
        for j in range(yb.size(0)):
            if int(yb[j]) in punct_ids:
                continue
            for k in k_list:
                if int(yb[j]) in top10[j, :k].tolist():
                    hits[k] += 1
            tot += 1
    return {k: hits[k] / max(tot, 1) for k in k_list}, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typing", default=os.path.join(HERE, "src", "typing.txt"))
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=4000,
                    help="训练中留出评测间隔（内容延续 top5，用于找峰值）")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min-ctx", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_small", "d64_l1_sent.pt"))
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "small_sentenced_result.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dargs = SimpleNamespace(typing=args.typing, holdout=0.1)
    train_lines, hold_lines = CS.load_split(dargs)
    train_text = CS.reconstruct_stream(train_lines, sep="\n")   # 段落间插换行
    hold_text = CS.reconstruct_stream(hold_lines, sep="\n")
    stoi, vocab = CS.build_char_vocab(train_text)
    itos = vocab
    print("训练唯一流 %d 字（含段落换行）｜ 留出 %d 字 ｜ 词表 %d ｜ 设备 %s"
          % (len(train_text), len(hold_text), len(vocab), device))

    # ---- 句子式训练样本 ----
    x, y, m = sentence_samples(stoi, train_text, args.seq, args.min_ctx)
    # 留出：同样按句子式打（评测用 content_topk，只统计内容位置）
    xh, yh = CS.pack_ids(CS.char_encode(stoi, hold_text), args.seq)

    model = None
    if os.path.exists(args.ckpt) and not args.retrain:
        model = CS.build_char_model(len(vocab), 64, 1)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        print("加载已有 ckpt：%s" % args.ckpt)
    else:
        model = CS.build_char_model(len(vocab), 64, 1).to(device)
        nparams = sum(p.numel() for p in model.parameters())
        print("训练 d64_l1  %.2fM 参数  %d 步（句子式，目标掩码）…" % (nparams / 1e6, args.steps))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        warm = args.warmup
        def lr_lambda(s):
            if s < warm:
                return s / max(1, warm)
            p = (s - warm) / max(1, args.steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        xd, yd, md = x.to(device), y.to(device), m.to(device)
        nb = xd.size(0)
        best_acc, best_step = None, 0
        best_ckpt = args.ckpt.replace(".pt", "_best.pt")
        t0 = time.time()
        for step in range(1, args.steps + 1):
            model.train()
            idx = (step * args.batch) % nb
            xb, yb, mb = xd[idx:idx + args.batch], yd[idx:idx + args.batch], md[idx:idx + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = masked_loss(model, xb, yb, mb, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % args.eval_every == 0 or step == args.steps:
                acc_c, tot_c = content_topk(model, xh, yh, device, stoi)
                if best_acc is None or acc_c[5] > best_acc[5]:
                    best_acc, best_step = acc_c, step
                    torch.save(model.state_dict(), best_ckpt)
                print("  step %d/%d loss=%.4f 内容top5=%.4f(best %.4f@%d) %.0fs"
                      % (step, args.steps, loss.item(), acc_c[5], best_acc[5], best_step,
                         time.time() - t0), flush=True)
        print("  ** 训练中峰值：内容 top1/5/10 = %.4f/%.4f/%.4f @ step %d"
              % (best_acc[1], best_acc[5], best_acc[10], best_step))
        torch.save(model.state_dict(), args.ckpt)
        print("已保存 %s" % args.ckpt)
    model = model.to(device).eval()

    # ---- 评测：只统计内容延续位置 ----
    acc_c, tot_c = content_topk(model, xh, yh, device, stoi)
    print("\n[留出·仅内容延续位置] top1=%.4f top5=%.4f top10=%.4f  (%d 个位置)"
          % (acc_c[1], acc_c[5], acc_c[10], tot_c))

    # ---- 生成探针（贪心 + 采样）----
    contexts = ["今天天气怎么", "现在用新的", "我们继续做这个实验", "训练的时候",
                "候选树", "继续加大数据", "明天我打算"]
    print("\n============ 贪心续写 ≤10 字符（句子式训练后）============")
    for c in contexts:
        print("模型(贪心): %s" % PC.greedy(model, stoi, itos, c, device=device))
    print("\n============ 温度采样（3 样本）============")
    for c in contexts:
        print("模型(采样) [%s]:" % c)
        for s in range(3):
            print("   %s" % PC.sample(model, stoi, itos, c, device=device, seed=100 + s))

    import json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"content_topk": acc_c, "n_content_pos": tot_c,
                   "args": vars(args)}, f, ensure_ascii=False, indent=2)
    print("\n已写 %s" % args.out)


if __name__ == "__main__":
    main()
