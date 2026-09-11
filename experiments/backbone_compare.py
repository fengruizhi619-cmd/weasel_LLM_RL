#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零样本对比：Base 骨干 vs Chat 骨干，不做任何训练。

两边各用自己的输出头（引擎会把 lm_head 与词嵌入解绑并转 fp32），
测同一批小说文本和同一批打字记录。用来回答"换成 chat 特化骨干会不会更好"。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "tools", "LlamaTreeExp")
sys.path.insert(0, EXP)

import gc
import torch
import probe_eval as pe
import unified_pipeline as up

NOVEL = r"C:\Users\Feng\Desktop\共享文件夹\银砂纪年\银砂纪年 第一卷 少女们，学院，龙与迫近的危机.txt"
MODELS = [("Base", r"E:\codex_data\研究\models\Qwen3-0.6B-Base"),
          ("Chat", r"E:\codex_data\研究\models\Qwen3-0.6B-Chat")]


def main():
    raw = pe.read_text(NOVEL)
    slices = [("小说内部(第1章附近)", raw[30:2030]),
              ("小说留出(第11章起2000字)", raw[24319:26319])]
    typing = pe.typing_probe(200)

    print("%-6s %-24s %-9s %-8s %-8s %-7s %-7s" %
          ("骨干", "切片", "前20累计p", "命中率", "脱靶率", "最长链", "top1"))
    for tag, path in MODELS:
        engine = up.TreeEngine(path, lr=1e-5, device=up.DEVICE, dtype="float16")
        for name, text in slices:
            r = pe.walk_metrics(engine, text)
            print("%-6s %-24s %-9.4f %-8.1f %-8.1f %-7d %-7.1f"
                  % (tag, name, r["mass20"], r["hit_rate"], r["miss_rate"],
                     r["longest"], r["hit1"]), flush=True)
        t = pe.score(engine, typing)
        print("%-6s %-24s 打字域 200 条：top1=%.1f%% top5=%.1f%% top20=%.1f%% MRR=%.3f 平均排名=%.0f"
              % (tag, "", t[0], t[1], t[2], t[3], t[4]), flush=True)
        print()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
