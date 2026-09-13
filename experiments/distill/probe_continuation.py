#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选树小模型输出探针：0.13M d64_l1 字符级模型，贪心续写 ≤10 字符。

数据/训练与 capacity_sweep 完全一致（同去重流、同词表、同种子、同配方 lr3e-4+warm150+3000步），
训练完保存 ckpt 并打印几个上下文（含用户示例"今天天气怎么"和几个真实域内上下文）的贪心续写。
顺带用字符 n-gram 做对照，直观展示"见过/没见过"的上下文两者的行为差异。

用法：
    python probe_continuation.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import capacity_sweep as CS  # noqa: E402


@torch.no_grad()
def greedy(model, stoi, itos, context, max_chars=10, device="cuda"):
    """贪心续写：每次取最后 256 字符窗口，argmax 下一个字符，追加。遇换行提前停。"""
    model.eval()
    ids = CS.char_encode(stoi, context)
    out = list(ids)
    u = stoi["<U>"]
    for _ in range(max_chars):
        win = out[-256:]
        x = torch.tensor([win], dtype=torch.long, device=device)
        logits = model(x).logits[0, -1]          # (V,)
        nxt = int(logits.argmax().item())
        if nxt == u:
            out.append(u)
            break
        out.append(nxt)
        if itos[nxt] == "\n":
            break
    gen = "".join(itos[i] for i in out[len(ids):])
    gen = gen.replace("\n", "⏎")
    return context + " → " + gen


@torch.no_grad()
def sample(model, stoi, itos, context, max_chars=10, device="cuda",
           temp=0.9, top_p=0.9, rep_penalty=1.15, seed=0):
    """温度采样 + top-p + 重复惩罚（候选树实际用的解码方式，而非贪心）。"""
    g = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    ids = CS.char_encode(stoi, context)
    out = list(ids)
    u = stoi["<U>"]
    for _ in range(max_chars):
        win = out[-256:]
        x = torch.tensor([win], dtype=torch.long, device=device)
        logits = model(x).logits[0, -1].clone()
        # 重复惩罚：已生成的字符降权
        if rep_penalty > 1.0:
            for t in out[len(ids):]:
                logits[t] /= rep_penalty
        logits = logits / max(temp, 1e-6)
        # top-p
        probs = torch.softmax(logits, dim=-1)
        sp, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=0)
        keep = cum <= top_p
        keep[0] = True
        probs[idx[~keep]] = 0.0
        probs = probs / probs.sum()
        nxt = int(torch.multinomial(probs, 1, generator=g).item())
        if nxt == u:
            out.append(u)
            break
        out.append(nxt)
        if itos[nxt] == "\n":
            break
    gen = "".join(itos[i] for i in out[len(ids):])
    return gen.replace("\n", "⏎")


def ngram_greedy(ng, context, max_chars=10):
    """n-gram 贪心：回退到最长见过的上下文。没见过的前缀则无法继续。"""
    out = context
    for _ in range(max_chars):
        ctx = out
        nxt = None
        for o in range(min(ng.order, len(ctx)), 0, -1):
            d = ng.cnt[o].get(ctx[-o:])
            if d:
                nxt = max(d, key=d.get)
                break
        if nxt is None:
            out += "⟦没见过的前缀，无法续写⟧"
            break
        if nxt == "\n":
            out += "⏎"
            break
        out += nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typing", default=os.path.join(HERE, "src", "typing.txt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_small", "d64_l1.pt"))
    ap.add_argument("--retrain", action="store_true", help="强制重训（默认有 ckpt 就直接加载）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 数据：与 capacity_sweep 完全一致 ----
    dargs = SimpleNamespace(typing=args.typing, holdout=0.1)
    train_lines, hold_lines = CS.load_split(dargs)
    train_text = CS.reconstruct_stream(train_lines)
    hold_text = CS.reconstruct_stream(hold_lines)
    stoi, vocab = CS.build_char_vocab(train_text)
    itos = vocab
    print("训练唯一流 %d 字 ｜ 词表 %d ｜ 设备 %s" % (len(train_text), len(vocab), device))

    # ---- 模型：d64_l1（甜点）----
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    model = None
    if os.path.exists(args.ckpt) and not args.retrain:
        model = CS.build_char_model(len(vocab), 64, 1)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        print("加载已有 ckpt：%s" % args.ckpt)
    else:
        model = CS.build_char_model(len(vocab), 64, 1)
        model = model.to(device)
        nparams = sum(p.numel() for p in model.parameters())
        print("训练 d64_l1  %.2fM 参数  %d 步…" % (nparams / 1e6, args.steps))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        warm = args.warmup
        def lr_lambda(s):
            if s < warm:
                return s / max(1, warm)
            p = (s - warm) / max(1, args.steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        x, y = CS.pack_ids(CS.char_encode(stoi, train_text), 256)
        x, y = x.to(device), y.to(device)
        nb = x.size(0)
        t0 = time.time()
        for step in range(1, args.steps + 1):
            model.train()
            idx = (step * 8) % nb
            xb, yb = x[idx:idx + 8], y[idx:idx + 8]
            opt.zero_grad(set_to_none=True)
            lg = model(xb).logits
            B, L, V = lg.shape
            loss = torch.zeros((), device=device)
            nch = 0
            for c in range(0, L, 128):
                loss = loss + F.cross_entropy(lg[:, c:c + 128].reshape(-1, V),
                                              yb[:, c:c + 128].reshape(-1))
                nch += 1
            (loss / nch).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 1000 == 0 or step == args.steps:
                print("  step %d/%d loss=%.4f %.0fs" % (step, args.steps, loss.item(), time.time() - t0))
        torch.save(model.state_dict(), args.ckpt)
        print("已保存 %s" % args.ckpt)
    model = model.to(device).eval()

    # ---- n-gram 对照 ----
    ng = CS.CharNGram(order=4)
    ng.fit(train_text)

    # ---- 贪心续写 ----
    contexts = [
        "今天天气怎么",                      # 用户示例（域外：语料是技术笔记）
        "现在用新的",                        # 域内（真实语料）
        "我们继续做这个实验",                  # 域内风格
        "训练的时候",                        # 域内
        "候选树",                            # 域内术语
        "继续加大数据",                      # 域内（语料尾部）
        "明天我打算",                        # 半域外
    ]
    print("\n============ 贪心续写 ≤10 字符 ============")
    for c in contexts:
        print("模型(贪心): %s" % greedy(model, stoi, itos, c, device=device))
    print("\n============ 温度采样（候选树实际解码方式，3 样本）============")
    for c in contexts:
        print("模型(采样) [%s]:" % c)
        for s in range(3):
            print("   %s" % sample(model, stoi, itos, c, device=device, seed=100 + s))
    print("\n-------- n-gram 对照（同样上下文）--------")
    for c in contexts:
        print("n-gram: %s" % ngram_greedy(ng, c))


if __name__ == "__main__":
    main()
