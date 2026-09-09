#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B benchmark: fp32 backbone vs fp16 backbone for RL update."""

import torch, time, sys, os
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
MODEL_PATH = (os.environ.get("WEASEL_LLM_MODEL", "").strip()
              or os.path.join(_REPO_ROOT, "models", "Qwen3-0.6B-Base"))
DEVICE = "cuda"
PROMPT = "明天天气如何，是"
LR = 1e-4


def load_and_prep(quantize):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    if model.lm_head.weight is model.model.embed_tokens.weight:
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.data.clone())
    for p in model.parameters():
        p.requires_grad = False
    model.lm_head.weight.requires_grad = True
    if quantize:
        model.model = model.model.half()
    optimizer = torch.optim.SGD([model.lm_head.weight], lr=LR)
    return model, tokenizer, optimizer


def get_hidden(model, input_ids):
    with torch.no_grad():
        h = model.model(input_ids=input_ids).last_hidden_state[0, -1, :]
        if h.dtype != torch.float32:
            h = h.float()
    return h


def get_top5(model, tokenizer, input_ids):
    with torch.no_grad():
        h = get_hidden(model, input_ids)
        logits = model.lm_head(h)
        probs = F.softmax(logits, dim=-1)
        top_p, top_id = torch.topk(probs, 5)
    return [(tokenizer.decode(top_id[i].item()), top_p[i].item()) for i in range(5)]


def rl_step(model, optimizer, h, target_id, reward):
    model.train()
    logits = model.lm_head(h)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -reward * log_probs[target_id]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    return loss.item()


def run_exp(quantize, label):
    print(f"\n{'='*50}\n  {label}\n{'='*50}")
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, optimizer = load_and_prep(quantize)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    mem = torch.cuda.memory_allocated() / 1048576
    print(f"  model mem: {mem:.0f}MB")

    before = get_top5(model, tokenizer, input_ids)
    print(f"  before: {[(t, f'{p:.4f}') for t,p in before[:3]]}")

    # get target
    h = get_hidden(model, input_ids)
    with torch.no_grad():
        logits = model.lm_head(h)
        probs = F.softmax(logits, dim=-1)
        top_p, top_id = torch.topk(probs, 1)
    target_id = top_id[0].item()
    reward = top_p[0].item()

    # bench forward x10
    def do_fwd():
        with torch.no_grad():
            for _ in range(10):
                model.model(input_ids=input_ids)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    do_fwd()
    torch.cuda.synchronize()
    fwd_ms = (time.monotonic() - t0) * 1000
    print(f"  fwd x10: {fwd_ms:.1f}ms ({fwd_ms/10:.1f}ms each)")

    # bench RL x10
    def do_rl():
        for _ in range(10):
            rl_step(model, optimizer, h, target_id, reward)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    do_rl()
    torch.cuda.synchronize()
    rl_ms = (time.monotonic() - t0) * 1000
    print(f"  RL x10: {rl_ms:.1f}ms ({rl_ms/10:.1f}ms each)")

    peak = torch.cuda.max_memory_allocated() / 1048576
    after = get_top5(model, tokenizer, input_ids)
    print(f"  after:  {[(t, f'{p:.4f}') for t,p in after[:3]]}")
    print(f"  peak mem: {peak:.0f}MB")

    del model, optimizer
    torch.cuda.empty_cache()
    return {"label": label, "mem": mem, "peak": peak, "fwd": fwd_ms, "rl": rl_ms}


print("[bench] GPU:", torch.cuda.get_device_name(0))
r = []
r.append(run_exp(False, "fp32 backbone"))
r.append(run_exp(True,  "fp16 backbone"))

print(f"\n{'='*50}\n  COMPARISON\n{'='*50}")
a, b = r[0], r[1]
for key, unit, fmt in [("mem","MB","{:.0f}"), ("peak","MB","{:.0f}"), ("fwd","ms","{:.1f}"), ("rl","ms","{:.1f}")]:
    va, vb = a[key], b[key]
    saved = (1 - vb/va) * 100 if va else 0
    print(f"  {key:6s}: {fmt.format(va)}{unit} -> {fmt.format(vb)}{unit} ({saved:.0f}% {'saved' if key in ('mem','peak') else 'faster'})")
