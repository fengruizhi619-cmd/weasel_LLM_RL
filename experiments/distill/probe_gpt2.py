#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPT2-Chinese 117M（uer/gpt2-chinese-cluecorpussmall）能力探针：
① 知识问答（续写式，看 117M 有没有事实知识）
② 生成质量（贪心/采样续写，与我们的 0.13M 字符模型对照）
③ 打字域留出 token 级 top-k（口径不同，仅参考）

用法：python probe_gpt2.py
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = r"E:\DSH_data\研究\models\GPT2-Chinese-117M"

from transformers import AutoTokenizer, GPT2LMHeadModel  # noqa: E402


@torch.no_grad()
def greedy(model, tok, prompt, max_new=12, device="cuda"):
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    new = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


@torch.no_grad()
def sample(model, tok, prompt, max_new=12, device="cuda", temp=0.9, top_p=0.9, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                         temperature=temp, top_p=top_p, pad_token_id=tok.eos_token_id,
                         generator=g)
    new = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print("GPT2-Chinese 参数 %.1fM ｜ 词表 %d ｜ 设备 %s\n" % (n / 1e6, len(tok), device))

    # ---- ① 知识问答 ----
    knowledge = [
        ("中国的首都是", "北京"),
        ("北京是中国的", "首都"),
        ("中国有56个", "民族"),
        ("一年有十二个", "月"),
        ("《红楼梦》的作者是", "曹雪芹"),
        ("床前明月光，疑是地上", "霜"),
        ("举头望明月，低头思", "故乡"),
        ("守株待", "兔"),
        ("抗日战争开始于1937年", "7月"),
        ("鲁迅的原名是", "周树人"),
        ("世界上最高的山峰是", "珠穆朗玛峰"),
        ("水的化学式是", "H2O"),
        ("人工智能的英文缩写是", "AI"),
    ]
    print("============ ① 知识问答（续写式；【期望】= 正确答案）============")
    hit = 0
    for p, exp in knowledge:
        out = greedy(model, tok, p, max_new=6)
        print("  %s → %s  【期望: %s】" % (p, out[:12], exp))
        if exp and exp in out[:12]:
            hit += 1
    print("  命中 %d/%d（粗判：期望串出现在续写前 12 字符里）" % (hit, len(knowledge)))

    # ---- ② 生成质量 ----
    contexts = ["今天天气怎么", "现在用新的", "我们继续做这个实验",
                "训练的时候", "候选树", "继续加大数据", "明天我打算"]
    print("\n============ ② 贪心续写 ≤12 字符 ============")
    for c in contexts:
        print("  %s → %s" % (c, greedy(model, tok, c).replace("\n", "⏎")))
    print("\n============ 温度采样（1 样本）============")
    for c in contexts[:4]:
        print("  [%s] %s" % (c, sample(model, tok, c).replace("\n", "⏎")))


if __name__ == "__main__":
    main()
