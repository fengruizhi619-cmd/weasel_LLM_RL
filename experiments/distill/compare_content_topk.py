#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公平对比：旧训练（全体位置）vs 句子式训练（掩码目标）在【仅内容延续位置】的 top-k。

旧 ckpt 的词表来自 sep="" 的流，新 ckpt 来自 sep="\\n" 的流，必须各自重建才能加载。
"""
import sys, os
from types import SimpleNamespace
import torch
import capacity_sweep as CS
import small_sentenced as SS

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

dargs = SimpleNamespace(typing=os.path.join(HERE, "src", "typing.txt"), holdout=0.1)
train_lines, hold_lines = CS.load_split(dargs)
device = "cuda" if torch.cuda.is_available() else "cpu"

def run(name, ckpt, sep):
    train_text = CS.reconstruct_stream(train_lines, sep=sep)
    hold_text = CS.reconstruct_stream(hold_lines, sep=sep)
    stoi, vocab = CS.build_char_vocab(train_text)
    model = CS.build_char_model(len(vocab), 64, 1)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()
    xh, yh = CS.pack_ids(CS.char_encode(stoi, hold_text), 256)
    acc, tot = SS.content_topk(model, xh, yh, device, stoi)
    print("%-10s 内容延续 top1/5/10 = %.4f / %.4f / %.4f  (%d 位置)"
          % (name, acc[1], acc[5], acc[10], tot))
    return acc

run("旧(全体位置)", os.path.join(HERE, "runs_small", "d64_l1.pt"), sep="")
run("新(句子式)", os.path.join(HERE, "runs_small", "d64_l1_sent.pt"), sep="\n")
