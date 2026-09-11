#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""骨干适配度对比：同样的语料、同样的训练流程，只换骨干，各训一个头。

    臂 Base  Qwen3-0.6B-Base 骨干（冻结）+ 自己原始的头 -> 混料训练 -> 存盘
    臂 Chat  Qwen3-0.6B-Chat 骨干（冻结）+ 自己原始的头 -> 混料训练 -> 存盘

两边完全一致的部分：小说第 1~10 章、打字池（含留出划分）、lr 1e-5、
总步数、打字比例 25%、随机种子。唯一变量是骨干。

比较的是"骨干的表征适不适合这个场景"：训练前各自零样本一行，
训练后各自一行，看差距是被抹平、拉大还是反转。
"""
import gc
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "tools", "LlamaTreeExp")
sys.path.insert(0, EXP)

import torch
import probe_eval as pe
import train_mix as tm
import unified_pipeline as up
import unified_watcher as uw

NOVEL = r"C:\Users\Feng\Desktop\共享文件夹\银砂纪年\银砂纪年 第一卷 少女们，学院，龙与迫近的危机.txt"
BASE = r"E:\codex_data\研究\models\Qwen3-0.6B-Base"
CHAT = r"E:\codex_data\研究\models\Qwen3-0.6B-Chat"
START, END = 30, 24319
RATIO, LR, STEPS, SEED = 0.25, 1e-5, 17001, 12345
HOLDOUT = 200


def run(tag, model_path, slices, typing, pool_records):
    print("=" * 70)
    print("臂 %s   骨干 %s" % (tag, model_path), flush=True)
    engine = up.TreeEngine(model_path, lr=LR, device=up.DEVICE, dtype="float16")

    def measure(prefix):
        out = []
        for name, text in slices:
            r = pe.walk_metrics(engine, text)
            out.append(r)
            print("  %-8s %-24s 命中%5.1f%% 累计p%.4f top1%5.1f%% 链%3d"
                  % (prefix, name, r["hit_rate"], r["mass20"], r["hit1"],
                     r["longest"]), flush=True)
        t = pe.score(engine, typing)
        print("  %-8s %-24s top1%5.1f%% top5%5.1f%% MRR%.3f 平均排名%.0f"
              % (prefix, "打字域(留出200)", t[0], t[1], t[3], t[4]), flush=True)
        return out, t

    before = measure("训练前")

    raw = pe.read_text(NOVEL)
    novel_text = raw[START:END]
    t0 = time.time()
    hn, tn = tm.walk_steps(engine, novel_text)
    hs, tg = [], []
    for ctx, txt in pool_records:          # 两个臂共用同一份快照
        for h, t in tm.steps_for(engine, ctx, txt):
            hs.append(h)
            tg.append(t)
    ht, tt = hs, tg
    print("  数据：小说 %d 步、打字池 %d 条去重 -> %d 步（%.0f 秒）"
          % (len(tn), len(pool_records), len(tt), time.time() - t0), flush=True)
    hN = torch.stack(hn); tN = torch.tensor(tn, device=engine.device)
    hT = torch.stack(ht); tT = torch.tensor(tt, device=engine.device)
    del hn, ht
    gc.collect()

    W = engine.model.lm_head.weight
    rng = random.Random(SEED)
    order = list(range(len(tt)))
    rng.shuffle(order)
    ti = ni = 0
    loss_sum = 0.0
    engine.model.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        if rng.random() < RATIO:
            k = order[ti % len(order)]; ti += 1
            h, t = hT[k], tT[k]
        else:
            k = ni % len(tN); ni += 1
            h, t = hN[k], tN[k]
        logits = engine.model.lm_head(h.unsqueeze(0))[0]
        loss = -torch.log_softmax(logits, dim=-1)[t]
        engine.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        engine.optimizer.step()
        loss_sum += float(loss.detach())
    engine.model.eval()
    engine.optimizer.zero_grad(set_to_none=True)
    print("  训练：%d 步（小说 %d / 打字 %d = %.1f%%），平均 loss %.4f，%.0f 秒"
          % (STEPS, ni, ti, 100.0 * ti / STEPS, loss_sum / STEPS, time.time() - t0),
          flush=True)

    ckpt = uw.CheckpointManager(engine, os.path.join(EXP, "diag", "checkpoints_mix_" + tag.lower()))
    ckpt.mark_dirty()
    ckpt.save_epoch(force=True, slots=("lm_head_t0.pt",))
    print("  已存盘 -> diag\\checkpoints_mix_%s" % tag.lower(), flush=True)

    after = measure("训练后")
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return before, after


def main():
    raw = pe.read_text(NOVEL)
    slices = [("小说内部(训练片段)", raw[30:2030]),
              ("小说留出(第11章起2000字)", raw[24319:26319])]
    typing = pe.typing_probe(HOLDOUT)
    # 打字池快照：读一次、切一次、去重一次，两个臂共用（否则采集器边跑边变）
    online, offline = tm.load_typing_sources()
    hold = online[-HOLDOUT:]
    seen, pool_records = set(), []
    for ctx, txt in online[:-HOLDOUT] + offline:
        if (ctx, txt) in seen:
            continue
        seen.add((ctx, txt))
        pool_records.append((ctx, txt))
    print("打字池快照：在线 %d 条（留出 %d 条评估）+ 离线 %d 条 -> 去重后 %d 条"
          % (len(online), len(hold), len(offline), len(pool_records)), flush=True)

    res = {}
    for tag, path in (("Base", BASE), ("Chat", CHAT)):
        res[tag] = run(tag, path, slices, typing, pool_records)
    print()
    print("=" * 70)
    print("汇总（同一批探针）")
    print("%-6s %-10s %-12s %-12s %-10s %-10s" %
          ("骨干", "阶段", "留出命中", "留出 top1", "打字 top1", "打字 top5"))
    for tag in ("Base", "Chat"):
        for i, stage in enumerate(("训练前", "训练后")):
            rows, t = res[tag][i]
            print("%-6s %-10s %-12.1f %-12.1f %-10.1f %-10.1f"
                  % (tag, stage, rows[1]["hit_rate"], rows[1]["hit1"], t[0], t[1]))


if __name__ == "__main__":
    main()
