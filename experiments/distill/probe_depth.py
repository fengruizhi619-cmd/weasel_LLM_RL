#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐层诊断：教师每一层的 logit-lens 读出到底能预测多准。

为什么必须先测这个：插入式蒸馏想在**每个循环边界**插一个监督信号，
边界的天然目标是"教师同深度处能给出的东西"。但深度就是干这个用的 ——
教师第 4 层的读出本来就是烂预测器。若把"匹配教师第 4 层读出"当全权重目标，
等于在教学生"早期输出要烂"。所以要先把这条曲线测出来：

    acc(d)  教师在第 d 层用「自己的 final norm + 自己的 lm_head」读出的 top1
    acc(28) 教师的真实水平（= 蒸馏上限）

读法：
  - acc(d) 接近 acc(28) 的深度区间 → 该处的 logit 目标可以给正常权重；
  - acc(d) 还在爬坡的深度区间  → 该处只适合给"表示层"目标（hint），
    或给很小的权重；给大权重会把学生早期读出按死在低水平上。

用法：
    python probe_depth.py --model <教师路径> --novel src/银砂纪年 第一卷.txt --typing src/typing.txt
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


@torch.no_grad()
def per_layer_acc(teacher, tok, text, seq, device, n_batch=6, batch=2):
    """返回 list：第 0..L 层的 logit-lens top1 准确率（0=嵌入输出）。

    **norm 的施加规则（踩过）**：transformers 的 `hidden_states` 约定是
    `[0..L-1]` 为各层**未归一化**的原始输出，`[L]` 是**已经过 final norm** 的。
    所以 d<L 要自己补 `model.norm`，d==L 千万不能再补（重复归一化会让
    argmax 明显变差 —— 实测 d=28 会从 0.325 掉到 0.284）。
    这里加了自检：d==L 的读数必须与直接 `teacher(ids).logits` 一致。
    """
    n_layer = teacher.config.num_hidden_layers
    acc = [0.0] * (n_layer + 1)
    chk = 0.0
    cnt = 0
    x, y = T.make_windows(tok, text, seq, device)
    nb = min(n_batch, x.size(0))
    for i in range(nb):
        ids = x[i:i + batch]
        yb = y[i:i + batch]
        if ids.numel() == 0:
            continue
        out = teacher(ids, output_hidden_states=True)
        hs = out.hidden_states           # 长度 L+1
        direct = out.logits.argmax(-1)
        for d, h in enumerate(hs):
            # x/y 已是 (chunk[:-1], chunk[1:])，所以 logits[j] 直接对 y[j]，
            # 不要再切 [:, :-1]（踩过：切了会 510 vs 512 不对齐）。
            hh = h if d == n_layer else teacher.model.norm(h)
            lg = teacher.lm_head(hh).argmax(-1)
            acc[d] += float((lg.reshape(-1) == yb.reshape(-1)).float().mean())
            if d == n_layer:
                chk += float((direct.reshape(-1) == lg.reshape(-1)).float().mean())
            del lg, hh
        del out
        cnt += 1
    self_check = chk / max(cnt, 1)
    if self_check < 0.9999:
        raise SystemExit("自检失败：最后一层 lens 与 teacher().logits 不一致（%.6f）—— "
                         "norm 施加规则错了" % self_check)
    return [a / max(cnt, 1) for a in acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "WEASEL_LLM_MODEL", r"E:\DSH_data\研究\models\Qwen3-0.6B-Base"))
    ap.add_argument("--novel", default="src/银砂纪年 第一卷.txt")
    ap.add_argument("--typing", default="src/typing.txt")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n-batch", type=int, default=6)
    ap.add_argument("--mix-typing", type=float, default=0.3)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--holdout-typing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260913)   # build_corpus 混料要用
    ap.add_argument("--out", default=os.path.join(HERE, "probe_depth_result.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    _, hold_novel, hold_typing = T.build_corpus(args)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()

    res = {}
    for name, text in (("novel", hold_novel), ("typing", hold_typing)):
        if not text:
            continue
        t0 = time.time()
        acc = per_layer_acc(teacher, tok, text, args.seq, device, args.n_batch, args.batch)
        res[name] = acc
        L = len(acc) - 1
        print("\n=== %s 留出集：逐层 logit-lens top1（%.0fs）===" % (name, time.time() - t0))
        print("%6s %8s %s" % ("深度", "acc", "曲线（每格 0.05，| = 最终层水平）"))
        final = acc[L]
        for d in range(L + 1):
            bar = "#" * int(round(acc[d] / 0.05))
            mark = "  <-- final" if d == L else ""
            print("%6d %8.4f %s%s" % (d, acc[d], bar, mark))

        # 各臂的循环边界 → 教师深度 → 该处目标的"质量"
        print("\n--- 各臂的监督点落在教师哪一层，以及那里的读出有多准 ---")
        for arm, (uniq, loops) in sorted(T.ARMS.items()):
            marks = []
            for t in range(1, loops + 1):
                d = min(uniq * t, L)
                marks.append("%d:%.3f%s" % (d, acc[d], "*" if uniq * t > L else ""))
            print("  %-8s U=%d T=%-2d 等效深度=%-3d  %s"
                  % (arm, uniq, loops, uniq * loops, " ".join(marks)))
        print("  （* = 超出教师深度，被夹到最后一层）")

    import json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "per_layer_acc": res}, f, ensure_ascii=False, indent=2)
    print("\n已写 %s" % args.out)


if __name__ == "__main__":
    main()
