#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选树小模型容量扫描：从 ~0.3M 往下（含）找"不再过拟合"的点。

用户方向：候选树不需要大模型；~1M 左右或更小即可，一直缩小看什么时候不过拟合。

关键架构调整：Qwen3 词表（151k）让嵌入层本身 ≥ 9.7M（151643×d），永远到不了 1M。
所以本脚本用**紧凑字符词表**（训练语料常用字 + ASCII + 标点 + UNK，约 8k），
嵌入降到 ~1M×d/8k 量级 —— 而且这正好和候选树/IEM 的字符级工作方式对齐。
字符级模型与字符级 n-gram 同口径直接可比（教师是 token 级，仅作参考）。

判据（用户要的"什么时候不过拟合"）：
    过拟合 = 训练 top5 高、留出 top5 低（差距大）；
    健康   = 留出 top5 高且接近训练（差距小）。
    报告每个尺寸：训练 top5(终) ｜ 留出 top5(训练中最好) ｜ 留出 top5(终) ｜ 差距。

用法：
    python capacity_sweep.py --steps 3000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 数据

def reconstruct_stream(lines, sep=""):
    """打字行是**渐进全量快照**（"现在用"→"现在用新的"，每行 = 到此刻为止的全文）。
    直接拼接会让同一段文本被喂几十遍（D0490 踩过），必须重建唯一文本流：
    若当前行以前一行开头，只取新增部分；否则另起一段（sep 不为空时在段间插入分隔符）。"""
    parts, prev = [], ""
    for ln in lines:
        if not ln:
            continue
        if prev and ln.startswith(prev):
            parts.append(ln[len(prev):])
        else:
            parts.append((sep if parts else "") + ln)
        prev = ln
    return "".join(parts)


def load_split(args):
    with open(args.typing, encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    k = max(1, int(len(lines) * args.holdout))
    return lines[:-k], lines[-k:]


def build_char_vocab(text, min_count=2, max_vocab=12000):
    cnt = Counter(text)
    base = ["\n"] + [chr(c) for c in range(32, 127)]
    have = set(base)
    common = [c for c, n in cnt.most_common() if c not in have and n >= min_count]
    common = common[:max_vocab - len(base) - 1]
    vocab = base + common + ["<U>"]
    stoi = {c: i for i, c in enumerate(vocab)}
    return stoi, vocab


def char_encode(stoi, text):
    u = stoi["<U>"]
    return [stoi.get(c, u) for c in text]


def pack_ids(ids, seq):
    n = (len(ids) - 1) // seq
    xs, ys = [], []
    for i in range(n):
        a = i * seq
        c = ids[a:a + seq + 1]
        xs.append(c[:-1])
        ys.append(c[1:])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


# ---------------------------------------------------------------- n-gram

class CharNGram:
    def __init__(self, order=4):
        self.order = order
        self.cnt = {}

    def fit(self, text):
        for o in range(1, self.order + 1):
            self.cnt.setdefault(o, {})
        for i in range(len(text) - 1):
            for o in range(1, self.order + 1):
                if i - o + 1 >= 0:
                    pre = text[i - o + 1:i + 1]
                    nxt = text[i + 1]
                    d = self.cnt[o].setdefault(pre, {})
                    d[nxt] = d.get(nxt, 0) + 1

    def _topk(self, ctx, k):
        for o in range(min(self.order, len(ctx)), 0, -1):
            d = self.cnt[o].get(ctx[-o:])
            if d:
                return [c for c, _ in sorted(d.items(), key=lambda x: -x[1])][:k]
        return []

    def eval_chars(self, hold_text, k_list=(1, 5, 10)):
        hits = {k: 0 for k in k_list}
        total = 0
        for i in range(len(hold_text) - 1):
            top = self._topk(hold_text[max(0, i - self.order + 1):i + 1], max(k_list))
            nxt = hold_text[i + 1]
            for k in k_list:
                if nxt in top[:k]:
                    hits[k] += 1
            total += 1
        return {k: hits[k] / max(total, 1) for k in k_list}


# ---------------------------------------------------------------- 模型

def build_char_model(vocab_size, hidden, layers):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    heads = 1 if hidden < 64 else (2 if hidden < 128 else 4)
    cfg = Qwen3Config(
        vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 4,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=heads, head_dim=hidden // heads,
        max_position_embeddings=4096, rope_theta=1e6, rms_norm_eps=1e-6,
        tie_word_embeddings=True, attention_dropout=0.0,
    )
    return Qwen3ForCausalLM(cfg)


class _LogitsOut:
    """统一 model(x).logits 接口：Qwen3ForCausalLM 返回 CausalLMOutput，循环块返回这个。"""

    def __init__(self, logits):
        self.logits = logits


class LoopedLM(torch.nn.Module):
    """循环 Transformer（Universal Transformer 式）：U=1 个共享层循环 rounds 次。
    循环的**价值是参数效率**（同有效深度下参数约 1/3，实测 D0112 只差 0.0024 AUC），
    所以"缩小 4 倍参数 + 循环补深度"在理论上是可行的组合。"""

    def __init__(self, vocab_size, hidden, rounds, rope_theta=1e6):
        super().__init__()
        from transformers import Qwen3Config, Qwen3Model
        heads = 1 if hidden < 64 else (2 if hidden < 128 else 4)
        self.heads = heads
        self.head_dim = hidden // heads
        self.rounds = rounds
        self.rope_theta = rope_theta
        cfg = Qwen3Config(
            vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 4,
            num_hidden_layers=1, num_attention_heads=heads,
            num_key_value_heads=heads, head_dim=self.head_dim,
            max_position_embeddings=4096, rope_theta=rope_theta, rms_norm_eps=1e-6,
            attention_dropout=0.0,
        )
        base = Qwen3Model(cfg)
        self.embed = base.embed_tokens
        self.layer = base.layers[0]          # 唯一共享层，循环 rounds 次
        self.round_emb = torch.nn.Embedding(rounds, hidden)
        torch.nn.init.zeros_(self.round_emb.weight)   # 零初始化起步，先不干扰
        self.norm = base.norm
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight       # tie 词嵌入

    def forward(self, input_ids):
        B, Ln = input_ids.shape
        dev = input_ids.device
        position_ids = torch.arange(Ln, device=dev).unsqueeze(0).expand(B, Ln)
        h = self.embed(input_ids)
        # 与 student_looped._rotary 同一算法：cos/sin 三维 (B,L,head_dim)，别补 heads 维
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=dev) / self.head_dim))
        freqs = torch.einsum("bl,d->bld", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(h.dtype), emb.sin().to(h.dtype)
        for t in range(self.rounds):
            h = h + self.round_emb(torch.full((B,), t, device=dev))[:, None, :]
            # Qwen3DecoderLayer 返回裸 tensor；mask 传 None 走 SDPA/GQA 内核
            h = self.layer(h, attention_mask=None, position_ids=position_ids,
                           position_embeddings=(cos, sin))
        return _LogitsOut(self.lm_head(self.norm(h)))


@torch.no_grad()
def topk_acc(model, x, y, device, k_list=(1, 5, 10), limit=None):
    model.eval()
    hits = {k: 0 for k in k_list}
    tot = 0
    nb = x.size(0) if limit is None else min(limit, x.size(0))
    for i in range(nb):
        lg = model(x[i:i + 1].to(device)).logits
        top10 = lg.topk(max(k_list), dim=-1).indices
        yb = y[i:i + 1].to(device)
        for k in k_list:
            hits[k] += (top10[:, :, :k] == yb.unsqueeze(-1)).any(-1).float().sum().item()
        tot += yb.numel()
    return {k: hits[k] / max(tot, 1) for k in k_list}


def vram(tag):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 2**20
        r = torch.cuda.memory_reserved() / 2**20
        print("  [显存] %-22s 已分配 %5.0fMB 已预留 %5.0fMB" % (tag, a, r), flush=True)


# 尺寸表：(名字, hidden, layers) —— 从 0.27M 到 17.7M，中心在 ~1M
SWEEP = [
    ("d32_l1", 32, 1),
    ("d64_l1", 64, 1),
    ("d64_l2", 64, 2),
    ("d128_l1", 128, 1),
    ("d128_l2", 128, 2),
    ("d256_l2", 256, 2),
    ("d256_l4", 256, 4),
    ("d512_l4", 512, 4),
]

# 循环变体：(名字, hidden, rounds) —— 参数与 dXX_l1 相同，但有效深度 ×rounds。
# 依据 D0112：循环 = 参数效率工具，同有效深度下循环约 1/3 参数、质量持平。
LOOP_SWEEP = [
    ("loop_d32_t4", 32, 4),    # 0.05M，有效深度 4 —— 比 plain d64_l1(0.13M) 小 2.6x
    ("loop_d32_t8", 32, 8),    # 0.05M，有效深度 8（D0112: 过深会轻微过拟合，作对照）
    ("loop_d64_t2", 64, 2),    # 0.13M，有效深度 2
    ("loop_d64_t4", 64, 4),    # 0.13M，有效深度 4
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typing", default=os.path.join(HERE, "src", "typing.txt"))
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8,
                    help="紧凑词表下 logits 很小（8×256×8k×4 ≈ 65MB），batch 8 安全")
    ap.add_argument("--steps", type=int, default=3000, help="每个尺寸的步数（统一，公平比较）")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--sizes", default="all",
                    help="逗号分隔的尺寸名（默认 all = 全部 SWEEP）")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", default=os.path.join(HERE, "capacity_sweep_result.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_lines, hold_lines = load_split(args)
    train_text = reconstruct_stream(train_lines)
    hold_text = reconstruct_stream(hold_lines)
    stoi, vocab = build_char_vocab(train_text)
    print("训练唯一流 %d 字 ｜ 留出唯一流 %d 字 ｜ 字符词表 %d" %
          (len(train_text), len(hold_text), len(vocab)))

    # n-gram 基线（字符级，同口径）
    ng = CharNGram(order=4)
    ng.fit(train_text)
    acc_ng = ng.eval_chars(hold_text)
    print("[基线] n-gram: top1=%.4f top5=%.4f top10=%.4f"
          % (acc_ng[1], acc_ng[5], acc_ng[10]))

    # 打包（字符级窗口）
    x, y = pack_ids(char_encode(stoi, train_text), args.seq)
    xh, yh = pack_ids(char_encode(stoi, hold_text), args.seq)
    print("训练窗口 %d ｜ 留出窗口 %d\n" % (x.size(0), xh.size(0)))

    if args.sizes == "all":
        sizes = [("plain",) + s for s in SWEEP] + [("loop",) + s for s in LOOP_SWEEP]
    else:
        want = set(x.strip() for x in args.sizes.split(","))
        sizes = ([("plain",) + s for s in SWEEP if s[0] in want] +
                 [("loop",) + s for s in LOOP_SWEEP if s[0] in want])
    rows = []
    for kind, name, a, b in sizes:
        if kind == "plain":
            model = build_char_model(len(vocab), a, b)
            eff = b
            tag = "d=%d L=%d 有效深度=%d" % (a, b, eff)
        else:
            model = LoopedLM(len(vocab), a, b)
            eff = b                       # 循环模型的有效深度 = 轮数（同一层应用几次）
            tag = "d=%d T=%d 有效深度=%d" % (a, b, eff)
        model = model.to(device)
        nparams = sum(p.numel() for p in model.parameters())
        print("=== %s  参数 %.2fM  %s ===" % (name, nparams / 1e6, tag), flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        warm = args.warmup
        def lr_lambda(s):
            if s < warm:
                return s / max(1, warm)
            p = (s - warm) / max(1, args.steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        xd, yd = x.to(device), y.to(device)
        nb = xd.size(0)
        best_hold = None
        t0 = time.time()
        for step in range(1, args.steps + 1):
            model.train()
            idx = (step * args.batch) % nb
            xb, yb = xd[idx:idx + args.batch], yd[idx:idx + args.batch]
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
                h5 = topk_acc(model, xh, yh, device, (5,))[5]
                if best_hold is None or h5 > best_hold:
                    best_hold = h5
                print("  step %d/%d loss=%.4f 留出top5=%.4f %.0fs"
                      % (step, args.steps, loss.item(), h5, time.time() - t0), flush=True)
        tr = topk_acc(model, x, y, device, (1, 5, 10))
        ho = topk_acc(model, xh, yh, device, (1, 5, 10))
        gap = tr[5] - ho[5]
        rows.append({"name": name, "kind": kind, "hidden": a, "b": b,
                     "eff_depth": eff, "params": nparams, "train": tr, "hold": ho,
                     "hold_top5_best": best_hold, "gap": gap})
        print("  训练 top1/5/10 = %.3f/%.3f/%.3f  ｜ 留出 = %.3f/%.3f/%.3f  ｜ 差距(top5)=%.3f"
              % (tr[1], tr[5], tr[10], ho[1], ho[5], ho[10], gap), flush=True)
        del model, opt, sched
        torch.cuda.empty_cache()
        vram("完成 %s" % name)

    # ---- 汇总表 ----
    print("\n================ 容量扫描汇总 ================")
    print("%-9s %8s %8s %8s %8s %8s" % ("尺寸", "参数", "训练top5", "留出top5", "留出best", "差距"))
    for r in rows:
        print("%-9s %7.2fM %8.3f %8.3f %8.3f %8.3f"
              % (r["name"], r["params"] / 1e6, r["train"][5], r["hold"][5],
                 r["hold_top5_best"], r["gap"]))
    print("n-gram 留出 top5 = %.4f（网络的对照线）" % acc_ng[5])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ngram": acc_ng, "vocab_size": len(vocab), "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print("\n已写 %s" % args.out)


if __name__ == "__main__":
    main()
