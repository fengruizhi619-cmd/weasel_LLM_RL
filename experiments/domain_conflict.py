#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打字域 vs 小说域：它们是方向冲突，还是正交？

对 head 分别求两个域的平均梯度，看夹角：
    cos > 0  两个目标同向，一起训是互相帮忙
    cos ~ 0  一阶互不干扰（正交），退化只能来自二阶漂移
    cos < 0  方向冲突，抢同一份参数，只能分开承载或混料

顺带量一下两个域的隐状态是不是挤在同一片区域——如果隐状态本来就分得开，
一个线性 head 原则上可以同时服务两边（不同区域用不同行）。

只读：不更新权重、不写文件。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "tools", "LlamaTreeExp")
sys.path.insert(0, EXP)

import torch
import torch.nn.functional as F
import probe_eval as pe
import unified_pipeline as up

NOVEL = r"C:\Users\Feng\Desktop\共享文件夹\银砂纪年\银砂纪年 第一卷 少女们，学院，龙与迫近的危机.txt"
N = 200


def main():
    engine = up.TreeEngine(up.MODEL_PATH, lr=up.LR, device=up.DEVICE, dtype="float16")
    engine.load_checkpoint(os.path.join(EXP, "diag", "checkpoints_online", "lm_head_t0.pt"))
    tok, dev = engine.tokenizer, engine.device
    W = engine.model.lm_head.weight

    typing = pe.typing_probe(N)
    novel = pe.text_probe(NOVEL, 24319, N)   # text_probe 收的是文件路径
    print("打字域样本 %d 条，小说域样本 %d 条" % (len(typing), len(novel)))

    def grads_and_hidden(data):
        W.grad = None
        hs = []
        for ctx, ch in data:
            ids = tok.encode(ctx, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = engine.model.model(input_ids=ids).last_hidden_state[0, -1, :]
            hs.append(h.float())
            logits = engine.model.lm_head(h.to(torch.float32).unsqueeze(0))[0]
            tid = int(tok.encode(ch, add_special_tokens=False)[0])
            loss = -F.log_softmax(logits, dim=-1)[tid]
            loss.backward()
        g = W.grad.detach().reshape(-1).clone()
        W.grad = None
        return g, torch.stack(hs)

    gT, hT = grads_and_hidden(typing)
    gN, hN = grads_and_hidden(novel)

    cos_gn = float(torch.dot(gT, gN) / (gT.norm() * gN.norm() + 1e-12))
    print()
    print("梯度夹角 cos(打字, 小说) = %+.4f   |g_打字| = %.1f  |g_小说| = %.1f"
          % (cos_gn, float(gT.norm()), float(gN.norm())))
    print("  含义：小说方向走一步，对打字损失的一阶影响 ∝ cos"
          "（%s）" % ("互相帮忙" if cos_gn > 0.05 else
                      "方向冲突" if cos_gn < -0.05 else "一阶几乎不干扰"))

    cT, cN = hT.mean(0), hN.mean(0)
    print()
    print("隐状态：域内平均余弦 %.3f（打字）/ %.3f（小说）"
          % (float(F.cosine_similarity(hT, cT.unsqueeze(0), dim=1).mean()),
             float(F.cosine_similarity(hN, cN.unsqueeze(0), dim=1).mean())))
    print("        两个域质心的余弦 %.3f，范数 %.1f / %.1f"
          % (float(F.cosine_similarity(cT.unsqueeze(0), cN.unsqueeze(0)).item()),
             float(cT.norm()), float(cN.norm())))
    cross = float(F.cosine_similarity(hT[:100].unsqueeze(1),
                                      hN[:100].unsqueeze(0), dim=2).mean())
    within = float(F.cosine_similarity(hT[:100].unsqueeze(1),
                                       hT[:100].unsqueeze(0), dim=2).mean())
    print("        跨域样本平均余弦 %.3f  vs  域内样本平均余弦 %.3f"
          % (cross, within))
    print()
    print("（只读：权重未改动，未写任何文件）")


if __name__ == "__main__":
    main()
