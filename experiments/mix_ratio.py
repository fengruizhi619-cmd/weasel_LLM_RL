#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混料比例对照：固定总步数，只变"打字样本 : 小说文本"的比例。

主干冻结，两边样本的隐状态只算一次（10000 字小说的全部位置 + 200 条打字样本），
之后每个臂只是 head 的前向/反向，跑完把权重还原，不落任何权重文件。

臂：打字占比 0% / 25% / 50% / 100%，每臂固定 STEPS 步，lr 固定 1e-4。

看两件事：小说留出文本的收益会不会被混料拖下来，打字域的退化会不会被拉住。
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "tools", "LlamaTreeExp")
sys.path.insert(0, EXP)

import torch
import corpus
import probe_eval as pe
import unified_pipeline as up

NOVEL = r"C:\Users\Feng\Desktop\共享文件夹\银砂纪年\银砂纪年 第一卷 少女们，学院，龙与迫近的危机.txt"
START, LEN = 30, 10000
CHUNK, PROMPT = 400, 64
STEPS, LR = 3000, 1e-5
RATIOS = [0.0, 0.25, 0.5, 1.0]
CKPT = os.path.join(EXP, "diag", "checkpoints_online", "lm_head_t0.pt")


def hidden_for(engine, contexts):
    """一批上下文的最后一个位置的隐状态（主干冻结，算一次就够）。"""
    tok, dev = engine.tokenizer, engine.device
    hs = []
    with torch.no_grad():
        for ctx in contexts:
            ids = tok.encode(ctx, return_tensors="pt").to(dev)
            hs.append(engine.model.model(
                input_ids=ids).last_hidden_state[0, -1, :].float())
    return torch.stack(hs)


def novel_positions(engine, text):
    """小说每个打分位置的隐状态和真值 token（标点跳过）。"""
    tok, dev = engine.tokenizer, engine.device
    hs, targets = [], []
    pos = 0
    while pos < len(text):
        head = text[max(0, pos - PROMPT):pos]
        if not head:
            pos += 1
            continue
        seg = text[pos:pos + CHUNK]
        ids = tok.encode(head + seg, add_special_tokens=False)
        n_head = len(tok.encode(head, add_special_tokens=False))
        pos += len(seg)
        if len(ids) <= n_head:
            continue
        with torch.no_grad():
            out = engine.model.model(input_ids=torch.tensor([ids], device=dev))
            h = out.last_hidden_state[0].float()
        for row in range(n_head, len(ids)):
            tid = ids[row]
            if up.is_no_target(tok.decode([tid])):
                continue
            hs.append(h[row - 1])
            targets.append(tid)
    return torch.stack(hs), torch.tensor(targets, device=dev)


def train_mix(engine, hN, tN, hT, tT, ratio, steps):
    W = engine.model.lm_head.weight
    engine.optimizer.param_groups[0]["lr"] = LR
    rng = random.Random(12345)
    order = list(range(len(hT)))
    rng.shuffle(order)
    ti = ni = 0
    engine.model.train()
    for _ in range(steps):
        if rng.random() < ratio:
            k = order[ti % len(order)]
            ti += 1
            h, t = hT[k], tT[k]
        else:
            k = ni % len(hN)
            ni += 1
            h, t = hN[k], tN[k]
        logits = engine.model.lm_head(h.unsqueeze(0))[0]
        loss = -torch.log_softmax(logits, dim=-1)[t]
        engine.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        engine.optimizer.step()
    engine.model.eval()
    engine.optimizer.zero_grad(set_to_none=True)
    return ti, ni


def main():
    raw = pe.read_text(NOVEL)
    train_text = raw[START:START + LEN]
    probe_in = raw[START:START + 2000]
    hold = raw[24319:26319]

    # 打字域划分：训练用前面的，评估用最后 200 条（两组不重叠，
    # 否则"打字变好"里混着"记住了训练样本"）
    recs = corpus.CorpusWriter(os.path.join(EXP, "diag", "corpus.jsonl")).read_all()
    acc = [(r["ctx"], r["typed"][0]) for r in recs
           if r.get("kind") == "accept" and r.get("ctx") and r.get("typed")]
    typing_train, typing = acc[:-200], acc[-200:]
    print("打字域：训练 %d 条（不重叠），评估 %d 条" % (len(typing_train), len(typing)))

    engine = up.TreeEngine(up.MODEL_PATH, lr=LR, device=up.DEVICE, dtype="float16")
    W = engine.model.lm_head.weight
    engine.load_checkpoint(CKPT)
    base = W.detach().clone()

    print("设置：总步数 %d、lr %g、比例 %s" % (STEPS, LR, RATIOS), flush=True)
    print("预计算隐状态 …", flush=True)
    t0 = time.time()
    hN, tN = novel_positions(engine, train_text)
    hT = hidden_for(engine, [c for c, _ in typing_train])
    tT = torch.tensor([int(engine.tokenizer.encode(c, add_special_tokens=False)[0])
                       for _c, c in typing_train], device=engine.device)
    print("  小说 %d 个位置，打字 %d 条，用时 %.0f 秒"
          % (len(tN), len(tT), time.time() - t0), flush=True)

    def measure(tag, p=None, n=None):
        r_in = pe.walk_metrics(engine, probe_in)
        r_hold = pe.walk_metrics(engine, hold)
        t = pe.score(engine, typing)
        with torch.no_grad():
            drift = float((W - base).pow(2).mean().sqrt()) / float(base.std())
        print("%-10s 小说内部 命中%5.1f%% 累计p%.4f 链%3d | 留出 命中%5.1f%% 累计p%.4f "
              "top1%5.1f%% 链%3d | 打字 top1%5.1f%% top5%5.1f%% | 位移%.3f%%"
              % (tag, r_in["hit_rate"], r_in["mass20"], r_in["longest"],
                 r_hold["hit_rate"], r_hold["mass20"], r_hold["hit1"],
                 r_hold["longest"], t[0], t[1], 100*drift), flush=True)
        return r_in, r_hold, t

    print()
    b_in, b_hold, b_typing = measure("训练前")
    for ratio in RATIOS:
        with torch.no_grad():
            W.copy_(base)
        t0 = time.time()
        ti, ni = train_mix(engine, hN, tN, hT, tT, ratio, STEPS)
        measure("打字%3d%%" % round(ratio * 100), ti, ni)
        print("    (%d 步：打字 %d / 小说 %d，用时 %.0f 秒)"
              % (STEPS, ti, ni, time.time() - t0), flush=True)
    with torch.no_grad():
        W.copy_(base)
    print()
    print("基线  小说内部 命中%.1f%% 累计p%.4f 链%d | 留出 命中%.1f%% 累计p%.4f top1%.1f%% 链%d | 打字 top1%.1f%% top5%.1f%%"
          % (b_in["hit_rate"], b_in["mass20"], b_in["longest"], b_hold["hit_rate"],
             b_hold["mass20"], b_hold["hit1"], b_hold["longest"], b_typing[0], b_typing[1]))
    print("权重已还原，未写盘")


if __name__ == "__main__":
    main()
