#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""折叠可行性探针：教师的 28 层里，哪些层是**可折叠**的？

背景（为什么必须先测这个）："改造原模型"（搬权重、不学语言）比"蒸馏一个新学生"的数据需求
低几个数量级 —— 基础能力本来就在权重里，我们只需要学"压缩造成的那点差异"。
但 naive 折叠已经测死：把教师前 U 层塞进共享块、循环 T 次，开局与教师的一致率是 **0.0000**
（教师第 1 层是为"接第 2 层"训练的，不是为"再接一次自己"）。

所以折叠方案不能靠假设（"每 2 层折一次"），得像查表一样问模型自己。本脚本测四件事：

  A) 基线：教师原样的 top1
  B) **逐层剪除代价**：把第 i 层替换成**精确恒等**（o_proj 与 down_proj 置零 —— 残差结构
     保证该层输出恒等于输入，已知 Qwen3 这两个投影无 bias），看掉多少。
     代价≈0 的层 = 冗余层，可以白捡压缩。
  C) **层间可互换矩阵** M[i][j]：把第 i 层的权重换成第 j 层的，看掉多少。
     M[i][j]≈基线 就说明"第 i 层干的事可以交给第 j 层"→ 那一对可以直接折成一层循环两次。
     **M[i][i] 必须精确等于基线**（自检，钉住实现正确）。
  D) **每层对残差流的改动幅度** ‖h_i − h_{i−1}‖ / ‖h_{i−1}‖，以及**线性 CKA** 矩阵
     —— 每层"干了多大的活"以及"层间表示有多像"，用来解释 B/C 的结果。

用法：
    python probe_foldable.py --model <教师> --domain typing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_distill as T  # noqa: E402


@torch.no_grad()
def acc_of(teacher, x, y, n):
    hit, tot = 0.0, 0
    for i in range(n):
        lg = teacher(x[i:i + 1]).logits
        hit += float((lg.argmax(-1) == y[i:i + 1]).float().sum())
        tot += y[i:i + 1].numel()
        del lg
    return hit / max(tot, 1)


def save_layer(layer):
    return {"o": layer.self_attn.o_proj.weight.detach().clone(),
            "d": layer.mlp.down_proj.weight.detach().clone(),
            "full": {k: v.detach().clone() for k, v in layer.state_dict().items()}}


@torch.no_grad()
def make_identity(layer):
    """把该层变成精确恒等：残差结构下 h ← h + Attn+MLP，两个输出投影置零即可。"""
    layer.self_attn.o_proj.weight.zero_()
    layer.mlp.down_proj.weight.zero_()


@torch.no_grad()
def restore_layer(layer, sd):
    layer.load_state_dict(sd["full"])


@torch.no_grad()
def cka_matrix(teacher, x, n, device):
    """线性 CKA：层间表示相似度（与 B/C 的功能探针互为解释）。"""
    L = teacher.config.num_hidden_layers
    feats = [[] for _ in range(L + 1)]
    for i in range(n):
        out = teacher(x[i:i + 1], output_hidden_states=True)
        for d, h in enumerate(out.hidden_states):
            feats[d].append(h.float().reshape(-1, h.size(-1)))
        del out
    feats = [torch.cat(f, 0) for f in feats]

    def cka(a, b):
        a = a - a.mean(0, keepdim=True)
        b = b - b.mean(0, keepdim=True)
        num = (a.T @ b).pow(2).sum()
        den = (a.T @ a).pow(2).sum().sqrt() * (b.T @ b).pow(2).sum().sqrt()
        return float(num / (den + 1e-9))

    M = [[0.0] * (L + 1) for _ in range(L + 1)]
    for i in range(L + 1):
        for j in range(i, L + 1):
            v = cka(feats[i], feats[j])
            M[i][j] = M[j][i] = v
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "WEASEL_LLM_MODEL", r"E:\DSH_data\研究\models\Qwen3-0.6B-Base"))
    ap.add_argument("--novel", default="src/银砂纪年 第一卷.txt")
    ap.add_argument("--typing", default="src/typing.txt")
    ap.add_argument("--domain", default="typing", choices=["typing", "novel"])
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--n-drop", type=int, default=4, help="剪层/幅度/CKA 用几个窗口")
    ap.add_argument("--n-matrix", type=int, default=2, help="互换矩阵用几个窗口（784 次前向）")
    ap.add_argument("--skip-matrix", action="store_true")
    ap.add_argument("--mix-typing", type=float, default=0.3)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--holdout-typing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out", default=os.path.join(HERE, "probe_foldable_result.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    _, hold_novel, hold_typing = T.build_corpus(args)
    text = hold_typing if args.domain == "typing" else hold_novel
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    layers = teacher.model.layers
    L = len(layers)
    x, y = T.make_windows(tok, text, args.seq, device)
    print("域=%s  窗口 %d 个（用 %d 做剪层/幅度，%d 做互换矩阵）"
          % (args.domain, x.size(0), args.n_drop, args.n_matrix))

    res = {}
    base = acc_of(teacher, x, y, args.n_drop)
    print("\n[A] 教师原样 top1 = %.4f" % base)
    res["baseline"] = base

    # 全恒等 = 相当于只到嵌入层（对照，用来确认"恒等"确实实现了）
    for l in layers:
        make_identity(l)
    all_id = acc_of(teacher, x, y, args.n_drop)
    # 恢复：重新载入整份权重最稳（比重放 state_dict 可靠）
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    print("    [自检] 全层恒等（≈只到嵌入层）top1 = %.4f" % all_id)
    if all_id >= base * 0.5:
        raise SystemExit("自检失败：全层恒等不该还有 %.4f（恒等没生效？）" % all_id)
    layers = teacher.model.layers
    if abs(acc_of(teacher, x, y, args.n_drop) - base) > 1e-9:
        raise SystemExit("自检失败：重载后基线对不上")

    # ---- B) 逐层剪除代价 ----
    print("\n[B] 逐层剪除代价（把该层换成精确恒等）")
    drop = []
    for i in range(L):
        sd = save_layer(layers[i])
        make_identity(layers[i])
        a = acc_of(teacher, x, y, args.n_drop)
        restore_layer(layers[i], sd)
        drop.append(base - a)
    res["drop_cost"] = drop
    order = sorted(range(L), key=lambda i: drop[i])
    print("    最冗余的 8 层（代价从低到高）：%s"
          % " ".join("%d:%.4f" % (i + 1, drop[i]) for i in order[:8]))
    print("    最关键的 8 层：%s"
          % " ".join("%d:%.4f" % (i + 1, drop[i]) for i in order[-8:]))
    # 一次性剪掉代价最小的若干层
    for k in (2, 4, 8):
        victim = order[:k]
        sds = {i: save_layer(layers[i]) for i in victim}
        for i in victim:
            make_identity(layers[i])
        a = acc_of(teacher, x, y, args.n_drop)
        for i in victim:
            restore_layer(layers[i], sds[i])
        print("    同时剪掉代价最小的 %2d 层 → top1 = %.4f（掉 %.4f）" % (k, a, base - a))

    # ---- D) 每层对残差流的改动幅度 + CKA ----
    print("\n[D] 每层对残差流的改动幅度 ‖Δh‖/‖h‖（用 %d 个窗口）" % args.n_drop)
    with torch.no_grad():
        mag = [0.0] * L
        cnt = 0
        for i in range(args.n_drop):
            out = teacher(x[i:i + 1], output_hidden_states=True)
            hs = out.hidden_states
            for d in range(1, L + 1):
                a, b = hs[d - 1].float(), hs[d].float()
                mag[d - 1] += float((b - a).norm() / (a.norm() + 1e-9))
            cnt += 1
            del out
        mag = [m / max(cnt, 1) for m in mag]
        del hs
    res["update_mag"] = mag
    for s in range(0, L, 14):
        print("    " + " ".join("%2d:%.2f" % (i + 1, mag[i]) for i in range(s, min(s + 14, L))))
    cka = cka_matrix(teacher, x, args.n_drop, device)
    res["cka"] = cka
    print("    CKA（与第 28 层的相似度）：" + " ".join("%d:%.2f" % (i + 1, cka[i][L]) for i in range(0, L, 2)))

    # ---- C) 层间可互换矩阵 ----
    if not args.skip_matrix:
        print("\n[C] 层间可互换矩阵 M[i][j]：把第 i 层换成第 j 层（784 次前向，用 %d 个窗口）"
              % args.n_matrix)
        t0 = time.time()
        # **参照必须用同一批窗口**（踩过：base 用 n_drop 个窗口、矩阵用 n_matrix 个，
        # 于是 M[i][i] 不为 0，自检直接报 0.0098 = 5/512 个 token）。
        base_m = acc_of(teacher, x, y, args.n_matrix)
        print("    矩阵参照（同一批 %d 窗口）base = %.4f" % (args.n_matrix, base_m))
        M = [[0.0] * L for _ in range(L)]
        full = {j: {k: v.detach().clone() for k, v in layers[j].state_dict().items()}
                for j in range(L)}
        for i in range(L):
            sdi = {k: v.detach().clone() for k, v in layers[i].state_dict().items()}
            for j in range(L):
                layers[i].load_state_dict(full[j])
                M[i][j] = base_m - acc_of(teacher, x, y, args.n_matrix)
            layers[i].load_state_dict(sdi)
            if (i + 1) % 7 == 0:
                print("      %d/%d  %.0fs" % (i + 1, L, time.time() - t0), flush=True)
        # 自检：M[i][i] 必须精确为 0
        off = max(abs(M[i][i]) for i in range(L))
        if off > 1e-9:
            raise SystemExit("自检失败：M[i][i] 应为 0，实测最大 %.6f" % off)
        print("    [自检] M[i][i] 全为 0 ✓")
        res["interchange"] = M
        print("\n    行=被替换的层，列=用来替换的层；数字=top1 掉多少（. = 不掉，x = 掉>0.2）")
        print("        " + "".join("%4d" % (j + 1) for j in range(L)))
        for i in range(L):
            row = "".join("%4s" % ("." if M[i][j] < 0.02 else ("x" if M[i][j] > 0.2 else "o"))
                          for j in range(L))
            print("    %3d %s" % (i + 1, row))
        # 最可互换的对
        pairs = sorted(((M[i][j], i + 1, j + 1) for i in range(L) for j in range(L) if i != j))
        print("\n    最可互换的 12 对（代价从低到高）：%s"
              % " ".join("%d←%d:%.3f" % (i, j, c) for c, i, j in pairs[:12]))
        # 相邻层互换代价（决定"渐进折叠"能不能逐步走）
        adj = [M[i][i + 1] for i in range(L - 1)]
        print("    相邻层互换代价（i+1←i）：%s"
              % " ".join("%d:%.3f" % (i + 1, adj[i]) for i in range(0, L - 1, 2)))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "n_layer": L, **res}, f, ensure_ascii=False, indent=2)
    print("\n已写 %s" % args.out)


if __name__ == "__main__":
    main()
