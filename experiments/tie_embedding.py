#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把训练好的 lm_head 同时当作输入侧的 embed_tokens，看效果变好还是变坏。

当前链路：输入侧用原始 embed_tokens（冻结），输出侧用训练过的 lm_head（fp32）。
本实验：把 lm_head 也灌进 embed_tokens，让两边共用同一份矩阵，再测同一批探针。

顺便量两份矩阵差多少——因为 head 是从 embed_tokens 复制出来的，差得越小，
这次替换就越接近"没变"。

只读：不动线上头，不写任何权重文件。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "tools", "LlamaTreeExp")
sys.path.insert(0, EXP)

import torch
import probe_eval as pe
import unified_pipeline as up

NOVEL = r"C:\Users\Feng\Desktop\共享文件夹\银砂纪年\银砂纪年 第一卷 少女们，学院，龙与迫近的危机.txt"
HEADS = [("最新混料头(mix_base)", os.path.join(EXP, "diag", "checkpoints_mix_base", "lm_head_t0.pt")),
         ("线上头(online 4713)", os.path.join(EXP, "diag", "checkpoints_online", "lm_head_t0.pt"))]


def main():
    raw = pe.read_text(NOVEL)
    slices = [("小说内部", raw[30:2030]), ("小说留出(第11章起2000字)", raw[24319:26319])]
    typing = pe.typing_probe(200)

    engine = up.TreeEngine(up.MODEL_PATH, lr=1e-5, device=up.DEVICE, dtype="float16")
    dev = engine.device
    emb = engine.model.model.embed_tokens.weight
    orig_emb = emb.detach().clone()
    W = engine.model.lm_head.weight
    print("输入侧 embed_tokens: %s  输出侧 lm_head: %s" % (tuple(emb.shape), tuple(W.shape)))
    print()

    def measure(tag):
        out = []
        for name, text in slices:
            r = pe.walk_metrics(engine, text)
            out.append(r)
            print("  %-22s %-22s 命中%5.1f%% 累计p%.4f top1%5.1f%% 链%3d"
                  % (tag, name, r["hit_rate"], r["mass20"], r["hit1"], r["longest"]), flush=True)
        t = pe.score(engine, typing)
        print("  %-22s %-22s top1%5.1f%% top5%5.1f%% MRR%.3f" % (tag, "打字域(留出200)", t[0], t[1], t[3]), flush=True)
        return out, t

    for tag, path in HEADS:
        m = torch.load(path, map_location=dev, weights_only=False)["lm_head_weight"].float()
        with torch.no_grad():
            W.copy_(m)
        d = m - orig_emb.float()
        rel = float(d.pow(2).mean().sqrt()) / float(orig_emb.float().std())
        cos = torch.nn.functional.cosine_similarity(m, orig_emb.float(), dim=1)
        print("== %s ==" % tag)
        print("  与原始 embed_tokens 的差: RMS %.3e = %.3f%% of std；逐行余弦 平均%.4f 最小%.4f 最大%.4f"
              % (float(d.pow(2).mean().sqrt()), 100*rel,
                 float(cos.mean()), float(cos.min()), float(cos.max())), flush=True)
        before = measure("A 原始输入+训练头")
        with torch.no_grad():
            emb.copy_(m)                      # 两边共用同一份矩阵
        after = measure("B 两边都用训练头")
        with torch.no_grad():
            emb.copy_(orig_emb)               # 还原输入侧
        b, a = before[0][1], after[0][1]
        bt, at = before[1], after[1]
        print("  变化（两侧相比单侧）: 留出命中 %+.1f 点  top1 %+.1f  打字 top1 %+.1f"
              % (a["hit_rate"]-b["hit_rate"], a["hit1"]-b["hit1"], at[0]-bt[0]), flush=True)
        print()


if __name__ == "__main__":
    main()
