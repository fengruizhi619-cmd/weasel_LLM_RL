#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学习率对照：同一段 1w 文本、同一步数、同一起点，只改 lr。

预测：如果"练坏"只是 lr 太大，那么降低 lr 会同时缩小收益和伤害，两者的
比例应该保持不变；如果比例明显改善，说明问题在目标函数而不在步长。

每个臂全程在显存里跑，跑完把权重还原；不落任何权重文件。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "tools", "LlamaTreeExp")
sys.path.insert(0, EXP)

import torch
import probe_eval as pe
import unified_pipeline as up

NOVEL = r"C:\Users\Feng\Desktop\共享文件夹\银砂纪年\银砂纪年 第一卷 少女们，学院，龙与迫近的危机.txt"
START, LEN = 30, 10000
CHUNK, PROMPT, K = 400, 64, 20
LRS = [1e-4, 3e-5, 1e-5, 3e-6]
CKPT = os.path.join(EXP, "diag", "checkpoints_online", "lm_head_t0.pt")


def train(engine, text, lr):
    tok, dev = engine.tokenizer, engine.device
    engine.optimizer.param_groups[0]["lr"] = lr
    steps = 0
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
            out = engine.model.model(input_ids=torch.tensor([ids], device=dev),
                                     use_cache=True)
        engine.model.train()
        for row in range(n_head, len(ids)):
            tid = ids[row]
            if up.is_no_target(tok.decode([tid])):
                continue
            hidden = out.last_hidden_state[0, row - 1, :]
            logits = engine.model.lm_head(hidden.to(torch.float32).unsqueeze(0))[0]
            loss = -torch.log_softmax(logits, dim=-1)[tid]
            engine.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            engine.optimizer.step()
            steps += 1
        engine.model.eval()
        engine.optimizer.zero_grad(set_to_none=True)
    return steps


def main():
    raw = pe.read_text(NOVEL)
    train_text = raw[START:START + LEN]
    hold = raw[24319:26319]
    typing = pe.typing_probe(200)

    engine = up.TreeEngine(up.MODEL_PATH, lr=5e-4, device=up.DEVICE, dtype="float16")
    W = engine.model.lm_head.weight
    engine.load_checkpoint(CKPT)
    base = W.detach().clone()

    def measure(tag):
        r_in = pe.walk_metrics(engine, train_text)
        r_hold = pe.walk_metrics(engine, hold)
        t = pe.score(engine, typing)
        with torch.no_grad():
            drift = float((W - base).pow(2).mean().sqrt()) / float(base.std())
        print("%-9s 训练 命中%5.1f%% 累计p%.4f 链%3d | 留出 命中%5.1f%% 累计p%.4f "
              "top1%5.1f%% 链%3d | 打字 top1%5.1f%% top5%5.1f%% | 位移%.3f%%"
              % (tag, r_in["hit_rate"], r_in["mass20"], r_in["longest"],
                 r_hold["hit_rate"], r_hold["mass20"], r_hold["hit1"],
                 r_hold["longest"], t[0], t[1], 100*drift), flush=True)
        return r_in, r_hold, t

    print("列：训练/留出 的 命中率、前20累计概率、最长链；打字域 top1/top5；权重位移")
    b_in, b_hold, b_typing = measure("训练前")
    for lr in LRS:
        with torch.no_grad():
            W.copy_(base)
        t0 = time.time()
        steps = train(engine, train_text, lr)
        print("  (%g 训练 %d 步 / %.0f 秒)" % (lr, steps, time.time() - t0), flush=True)
        measure("lr=%.0e" % lr)
    with torch.no_grad():
        W.copy_(base)
    print()
    print()
    print("基线  训练 命中%.1f%% 累计p%.4f 链%d | 留出 命中%.1f%% 累计p%.4f top1%.1f%% 链%d | 打字 top1%.1f%% top5%.1f%%"
          % (b_in["hit_rate"], b_in["mass20"], b_in["longest"],
             b_hold["hit_rate"], b_hold["mass20"], b_hold["hit1"],
             b_hold["longest"], b_typing[0], b_typing[1]))
    print("参考 5e-4（上一轮单独跑）：训练 命中77.3%% 累计p0.9172 链41 | 留出 命中67.4%% 累计p0.9126 top1 35.6%% 链13 | 打字 top1 8.0%%")
    print("权重已还原，未写盘")


if __name__ == "__main__":
    main()
