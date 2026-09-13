#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小型网络候选树基线：同一个打字语料、同一个留出集、同一个 tokenizer，
同时测三个对象 —— 用户问"为什么非得用大模型"的直接回答：

  A) 字符 n-gram（4 阶回退）：零训练的地板。"候选树完全可以用其他方式做"的最轻形态。
  B) 小 Transformer（Qwen3 结构、hidden 256 × 4 层、约 50M，词表复用 Qwen3 tokenizer）：
     监督式 next-char 训练。小型网络 + 离线监督打底（在线 RL 是第二步）。
  C) 教师（Qwen3-0.6B）参考：top1 ≈ 0.82。

指标用 top-1 / top-5 / top-10：候选树是"多分支给用户挑"，top-k 比 top-1 更接近体验。
留出集与 train_distill 完全一致（typing.txt 尾部 10% 行），数字可直接对比。

用法：
    python small_predictor.py --steps 4000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_distill as T  # noqa: E402


# ---------------------------------------------------------------- 数据

def load_split(args):
    """typing.txt 按行：留出 = 尾部 10%（与 train_distill 的 holdout-typing 一致）。"""
    with open(args.typing, encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    k = max(1, int(len(lines) * args.holdout))
    train_lines, hold_lines = lines[:-k], lines[-k:]
    return train_lines, hold_lines


def pack_windows(tok, lines, seq):
    """把多行 token 打包成固定长窗口（行间以换行 token 分隔；超长行单独切）。"""
    ids = tok("\n".join(lines), add_special_tokens=False)["input_ids"]
    n = (len(ids) - 1) // seq
    xs, ys = [], []
    for i in range(n):
        a = i * seq
        c = ids[a:a + seq + 1]
        xs.append(c[:-1])
        ys.append(c[1:])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


# ---------------------------------------------------------------- n-gram 基线

class CharNGram:
    """4 阶回退计数模型（字符级）。预测给定前缀的下一字符 top-k。"""

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
            ctx = hold_text[max(0, i - self.order + 1):i + 1]
            top = self._topk(ctx, max(k_list))
            nxt = hold_text[i + 1]
            for k in k_list:
                if nxt in top[:k]:
                    hits[k] += 1
            total += 1
        return {k: hits[k] / max(total, 1) for k in k_list}


# ---------------------------------------------------------------- 小 Transformer

def build_small(tok, hidden=256, layers=4):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=tok.vocab_size, hidden_size=hidden, intermediate_size=hidden * 4,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=hidden // 4, max_position_embeddings=2048, rope_theta=1e6,
        rms_norm_eps=1e-6, tie_word_embeddings=True, attention_dropout=0.0,
    )
    m = Qwen3ForCausalLM(cfg)
    n = sum(p.numel() for p in m.parameters())
    print("小模型: hidden=%d layers=%d  参数=%s (%.1fM)" % (hidden, layers, f"{n:,}", n / 1e6))
    return m, cfg


@torch.no_grad()
def topk_acc(model, x, y, device, k_list=(1, 5, 10), max_batches=8):
    model.eval()
    hits = {k: 0 for k in k_list}
    tot = 0
    nb = min(max_batches, x.size(0))
    for i in range(nb):
        lg = model(x[i:i + 1].to(device)).logits          # fp16，不转 fp32
        top10 = lg.topk(max(k_list), dim=-1).indices
        yb = y[i:i + 1].to(device)
        for k in k_list:
            hits[k] += (top10[:, :, :k] == yb.unsqueeze(-1)).any(-1).float().sum().item()
        tot += yb.numel()
        del lg, top10
    return {k: hits[k] / max(tot, 1) for k in k_list}, tot


def vram(tag):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 2**20
        r = torch.cuda.memory_reserved() / 2**20
        free, _ = torch.cuda.mem_get_info()
        print("  [显存] %-22s 已分配 %5.0fMB 已预留 %5.0fMB 可用 %4.0fMB"
              % (tag, a, r, free / 2**20), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "WEASEL_LLM_MODEL", r"E:\DSH_data\研究\models\Qwen3-0.6B-Base"))
    ap.add_argument("--typing", default=os.path.join(HERE, "src", "typing.txt"))
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4,
                    help="显存纪律：词表 151k，fp32 logits 每样本 ~156MB，batch 4 峰值 ~1GB。"
                         "别调到 16 —— 会把 8GB 卡（还共享着 wallpaper/NVIDIA/llama-server 等）"
                         "挤到共享显存去。")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", default=os.path.join(HERE, "small_predictor_result.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    train_lines, hold_lines = load_split(args)
    train_text = "\n".join(train_lines)
    hold_text = "\n".join(hold_lines)
    print("训练 %d 行 / %d 字 ｜ 留出 %d 行 / %d 字"
          % (len(train_lines), len(train_text), len(hold_lines), len(hold_text)))

    res = {"args": vars(args), "holdout_chars": len(hold_text)}
    # ---- A) n-gram ----
    t0 = time.time()
    ng = CharNGram(order=4)
    ng.fit(train_text)
    acc_ng = ng.eval_chars(hold_text)
    print("\n[A] 字符 n-gram（4 阶回退）: top1=%.4f top5=%.4f top10=%.4f  (%.0fs)"
          % (acc_ng[1], acc_ng[5], acc_ng[10], time.time() - t0))
    res["ngram"] = acc_ng

    # ---- 打包窗口 ----
    x, y = pack_windows(tok, train_lines, args.seq)
    xh, yh = pack_windows(tok, hold_lines, args.seq)
    print("训练窗口 %d ｜ 留出窗口 %d" % (x.size(0), xh.size(0)))
    res["train_windows"] = int(x.size(0))
    res["hold_windows"] = int(xh.size(0))

    # ---- C) 教师参考（同一批留出窗口）----
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    vram("教师加载")
    acc_t, _ = topk_acc(teacher, xh, yh, device)
    print("[C] 教师: top1=%.4f top5=%.4f top10=%.4f" % (acc_t[1], acc_t[5], acc_t[10]))
    res["teacher"] = acc_t
    del teacher
    torch.cuda.empty_cache()
    vram("教师已释放")

    # ---- B) 小 Transformer ----
    small, cfg = build_small(tok)
    small = small.to(device)                 # fp32：避免 fp16 无梯度缩放导致的 NaN
    vram("小模型加载")
    opt = torch.optim.AdamW(small.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    x, y = x.to(device), y.to(device)
    n_batch = x.size(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        small.train()
        idx = (step * args.batch) % n_batch
        xb = x[idx:idx + args.batch]
        yb = y[idx:idx + args.batch]
        opt.zero_grad(set_to_none=True)
        lg = small(xb).logits               # fp16 (B,L,V)，不整张转 fp32
        B, L, V = lg.shape
        # 交叉熵按序列分块算：限制 fp32 内部上转换的瞬时占用
        loss = torch.zeros((), device=device)
        n_chunk = 0
        for c in range(0, L, 64):
            loss = loss + F.cross_entropy(lg[:, c:c + 64].reshape(-1, V),
                                          yb[:, c:c + 64].reshape(-1))
            n_chunk += 1
        loss = loss / n_chunk
        loss.backward()
        torch.nn.utils.clip_grad_norm_(small.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            print("  step %d/%d loss=%.4f lr=%.2e %.0fs"
                  % (step, args.steps, loss.item(), sched.get_last_lr()[0], time.time() - t0))
            vram("训练 step %d" % step)
    acc_s, tot = topk_acc(small, xh, yh, device)
    print("\n[B] 小 Transformer: top1=%.4f top5=%.4f top10=%.4f  (%d 个 token)"
          % (acc_s[1], acc_s[5], acc_s[10], tot))
    vram("训练结束")
    res["small"] = acc_s

    import json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n已写 %s" % args.out)


if __name__ == "__main__":
    main()
