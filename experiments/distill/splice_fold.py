#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""改造原模型：把教师的中间若干层**就地**换成"一个共享块循环 N 次"，零训练测准确率。

这是"改造"而不是"蒸馏"的最纯形式：权重全部来自教师本身（取均值 / 取中间某一层），
没有任何随机初始化，也没有任何训练。要回答的问题只有一个：

    把 [lo..hi] 这段层换成"一块共享权重循环 (hi-lo+1) 次"，掉多少分？
    再往下砍轮数（13 → 10 → 8 → 6 → 4 → 2），掉多少分？

依据（probe_foldable.py 实测的层间可互换矩阵）：
  层 1~2 不可替代（换掉掉 >0.2）；层 4~16 高度可互换（互相换掉 <0.02，多为负值）；
  层 17~27 半可互换（0.02~0.2）；层 28 独特。
所以"可折叠的候选"就是中段那两块，本脚本把它们逐一折掉试试。

注意：**折叠共享只省参数（权重内存），不省层前向数**。要省算力必须砍轮数 ——
所以本脚本的核心产出是"轮数 → 准确率"这条曲线，它同时给出参数与算力的收益边界。

用法：
    python splice_fold.py --model <教师> --band 4-16
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_distill as T  # noqa: E402


def rotary(cfg, h, position_ids):
    """与 student_looped._rotary 同一套算法（三维 (B,L,head_dim)，不要自己补 heads 维）。"""
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    inv_freq = 1.0 / (cfg.rope_theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=h.device) / head_dim))
    freqs = torch.einsum("bl,d->bld", position_ids.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(h.dtype), emb.sin().to(h.dtype)


class Spliced(torch.nn.Module):
    """教师 = pre(若干原层) + mid(若干共享块循环) + post(其余原层) + norm + head。

    全部权重都来自教师；没有任何随机初始化。
    init="mean" 取各块覆盖层的权重均值（等价 model soup），"pick" 取该块正中间那一层。
    blocks=k：把这段层按顺序切成 k 段，每段一个共享块，循环时按"粗粒度深度"轮流使用
    （k=1 就是全段一块）。这是在"1 块（最省）"和"13 层（最准）"之间的折中轴。
    mode="cut"：整段换成恒等（删掉但不循环），作为"这段层总共有多重要"的参照 ——
    没有这个参照，折叠掉的分数无法归因（是删掉了层，还是重复同一块的漂移？）。
    """

    def __init__(self, teacher, lo, hi, init="mean", blocks=1, mode="fold"):
        super().__init__()
        self.t = teacher
        self.lo, self.hi = lo, hi          # 1 基，闭区间；[lo..hi] 被折叠/剪除
        self.mode = mode
        self.pre = teacher.model.layers[:lo - 1]
        self.post = teacher.model.layers[hi:]
        src = list(range(lo - 1, hi))
        self.n_src = len(src)
        self.blocks = []
        if mode == "fold":
            k = max(1, min(blocks, self.n_src))
            groups = [src[i * self.n_src // k:(i + 1) * self.n_src // k] for i in range(k)]
            for g in groups:
                blk = copy.deepcopy(teacher.model.layers[g[0]])
                if init == "mean" and len(g) > 1:
                    sds = [teacher.model.layers[i].state_dict() for i in g]
                    with torch.no_grad():
                        for key in blk.state_dict():
                            blk.state_dict()[key].copy_(
                                torch.stack([s[key].float() for s in sds]).mean(0))
                elif init == "pick":
                    blk = copy.deepcopy(teacher.model.layers[g[len(g) // 2]])
                self.blocks.append(blk)
            self.k = k

    @torch.no_grad()
    def forward(self, ids, rounds=None):
        cfg = self.t.config
        B, Ln = ids.shape
        dev = ids.device
        pid = torch.arange(Ln, dtype=torch.long, device=dev).unsqueeze(0).expand(B, Ln)
        h = self.t.model.embed_tokens(ids)
        cos, sin = rotary(cfg, h, pid)
        for layer in self.pre:
            h = layer(h, position_ids=pid, position_embeddings=(cos, sin))
        if self.mode == "cut":
            pass                               # 整段剪掉：恒等
        else:
            r = self.n_src if rounds is None else rounds
            for t in range(r):
                idx = min(self.k - 1, (t * self.k) // max(1, r))   # 粗粒度：把轮次分成 k 段
                h = self.blocks[idx](h, position_ids=pid, position_embeddings=(cos, sin))
        for layer in self.post:
            h = layer(h, position_ids=pid, position_embeddings=(cos, sin))
        return self.t.lm_head(self.t.model.norm(h))


@torch.no_grad()
def acc(model, x, y, n, rounds=None):
    hit, tot = 0.0, 0
    for i in range(n):
        out = model(x[i:i + 1]) if rounds is None else model(x[i:i + 1], rounds)
        lg = out.logits if hasattr(out, "logits") else out   # 教师返回对象，Spliced 返回张量
        hit += float((lg.argmax(-1) == y[i:i + 1]).float().sum())
        tot += y[i:i + 1].numel()
        del lg, out
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "WEASEL_LLM_MODEL", r"E:\DSH_data\研究\models\Qwen3-0.6B-Base"))
    ap.add_argument("--novel", default="src/银砂纪年 第一卷.txt")
    ap.add_argument("--typing", default="src/typing.txt")
    ap.add_argument("--domain", default="typing", choices=["typing", "novel"])
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--band", default="4-16", help="被折叠的层区间（1 基，闭），如 4-16")
    ap.add_argument("--mix-typing", type=float, default=0.3)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--holdout-typing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out", default=os.path.join(HERE, "splice_fold_result.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    _, hold_novel, hold_typing = T.build_corpus(args)
    text = hold_typing if args.domain == "typing" else hold_novel
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    L = teacher.config.num_hidden_layers
    x, y = T.make_windows(tok, text, args.seq, device)
    lo, hi = (int(v) for v in args.band.split("-"))
    assert 1 <= lo <= hi <= L, "band 越界"
    base = acc(teacher, x, y, args.n)
    print("域=%s  窗口 %d 个（用 %d）" % (args.domain, x.size(0), args.n))
    print("[基线] 教师原样 top1 = %.4f" % base)

    res = {"baseline": base, "band": [lo, hi], "runs": {}}

    # 参照：整段剪掉（恒等，不循环）—— 折叠掉的分数要靠它归因
    cut = Spliced(teacher, lo, hi, mode="cut").to(device).eval()
    a_cut = acc(cut, x, y, args.n)
    print("\n[参照·整段剪除为恒等] [%d..%d] 全删 → top1 = %.4f（掉 %.4f）"
          % (lo, hi, a_cut, base - a_cut))
    res["cut"] = a_cut
    del cut
    torch.cuda.empty_cache()

    for blocks in (1, 2, 3):
        if blocks > (hi - lo + 1):
            continue
        for init in ("mean", "pick"):
            m = Spliced(teacher, lo, hi, init=init, blocks=blocks).to(device).eval()
            n_full = m.n_src
            print("\n=== 折叠 [%d..%d]（%d 层 → %d 块共享，init=%s）==="
                  % (lo, hi, n_full, blocks, init))
            print("%8s %10s %10s" % ("轮数", "top1", "掉分"))
            row = {}
            for r in sorted({n_full, max(1, n_full - 3), max(1, n_full // 2), 4, 2, 1}, reverse=True):
                if r > n_full:
                    continue
                a = acc(m, x, y, args.n, rounds=r)
                row[str(r)] = a
                print("%8d %10.4f %10.4f" % (r, a, base - a))
            res["runs"]["b%d_%s" % (blocks, init)] = row
            del m
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), **res}, f, ensure_ascii=False, indent=2)
    print("\n已写 %s" % args.out)


if __name__ == "__main__":
    main()
