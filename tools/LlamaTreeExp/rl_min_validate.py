#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal RL validation: freeze backbone, train only lm_head via
confidence-weighted reward from candidate tree character matching."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

MODEL_PATH = r"E:\codex_data\研究\models\Qwen3-0.6B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4  # small lr to avoid catastrophic forgetting
TOP_N = 5

def get_top_k(model, tokenizer, input_ids, k=5):
    """Query model for top-k next token predictions."""
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1, :]  # last position
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k)
    results = []
    for i in range(top_ids.shape[0]):
        tok_str = tokenizer.decode(top_ids[i].item())
        results.append({
            "token_id": top_ids[i].item(),
            "token": tok_str,
            "p": top_probs[i].item(),
        })
    return results

def show_candidates(results, label=""):
    print(f"\n  [{label}] top {len(results)} candidates:")
    for i, r in enumerate(results):
        print(f"    {i+1}. p={r['p']:.4f} token={r['token']!r}")

def main():
    print(f"[rl-min] loading model from {MODEL_PATH}")
    print(f"[rl-min] device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)

    # [STEP 1] Check weight tying and untie if needed
    if model.lm_head.weight is model.model.embed_tokens.weight:
        print("[rl-min] lm_head tied to embed_tokens, untying...")
        import torch.nn as nn
        old_embed = model.model.embed_tokens.weight.data.clone()
        model.lm_head.weight = nn.Parameter(old_embed.clone())
        print("[rl-min] lm_head untied (independent copy created)")

    # Freeze everything except lm_head
    for param in model.parameters():
        param.requires_grad = False
    model.lm_head.weight.requires_grad = True

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[rl-min] total params: {total_params:,}")
    print(f"[rl-min] trainable: {trainable_params:,} ({trainable})")

    optimizer = torch.optim.SGD([model.lm_head.weight], lr=LR)

    # [STEP 2] Prompt
    prompt = "明天天气如何，是"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    print(f"\n[rl-min] prompt: {prompt!r}")
    print(f"[rl-min] input_ids shape: {input_ids.shape}")

    # [STEP 3] Query BEFORE update
    results_before = get_top_k(model, tokenizer, input_ids, TOP_N)
    show_candidates(results_before, "BEFORE update")

    # [STEP 4] Simulate user typing: user picks "晴" (which is in candidates)
    # Find the token that starts with 晴
    user_char = "晴"
    target = None
    reward = 0.0
    for r in results_before:
        tok = r["token"]
        if tok.startswith(user_char):
            match_ratio = len(user_char) / len(tok)
            reward = r["p"] * match_ratio
            target = r["token_id"]
            print(f"\n[rl-min] user typed: {user_char!r}")
            print(f"[rl-min] matched token: {tok!r} (match_ratio={match_ratio:.2f})")
            print(f"[rl-min] reward = p({r['p']:.4f}) x ratio({match_ratio:.2f}) = {reward:.4f}")
            break

    if target is None:
        # Check if user_char is part of any multi-char token
        for r in results_before:
            tok = r["token"]
            if user_char in tok:
                idx = tok.index(user_char)
                match_ratio = (idx + 1) / len(tok)
                reward = r["p"] * match_ratio
                target = r["token_id"]
                print(f"\n[rl-min] user typed: {user_char!r}")
                print(f"[rl-min] partial match in token: {tok!r} at pos {idx}")
                print(f"[rl-min] reward = p({r['p']:.4f}) x ratio({match_ratio:.2f}) = {reward:.4f}")
                break

    if target is None:
        print(f"\n[rl-min] user char {user_char!r} not in candidates, skip update")
        return

    # [STEP 5] Gradient ascent on lm_head (reinforce the correct token)
    print(f"\n[rl-min] doing gradient step on lm_head (lr={LR})")
    model.train()
    out = model(input_ids)
    logits = out.logits[0, -1, :]  # last position logits

    # Weighted cross-entropy: increase prob of target token, weighted by reward
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -reward * log_probs[target]
    loss.backward()

    # Check gradient only exists on lm_head
    grad_norm = model.lm_head.weight.grad.norm().item()
    print(f"[rl-min] loss={loss.item():.6f} lm_head_grad_norm={grad_norm:.6f}")

    optimizer.step()
    optimizer.zero_grad()
    model.eval()

    # [STEP 6] Query AFTER update
    results_after = get_top_k(model, tokenizer, input_ids, TOP_N)
    show_candidates(results_after, "AFTER update")

    # [STEP 7] Compare: did the target token's probability increase?
    print(f"\n[rl-min] === COMPARISON ===")
    before_map = {r["token_id"]: r["p"] for r in results_before}
    after_map = {r["token_id"]: r["p"] for r in results_after}

    # Target token probability change
    p_before = before_map.get(target, 0.0)
    p_after = after_map.get(target, 0.0)
    delta = p_after - p_before
    direction = "↑ INCREASED" if delta > 0 else "↓ DECREASED" if delta < 0 else "→ SAME"
    print(f"  target token {tokenizer.decode(target)!r}: "
          f"p {p_before:.4f} → {p_after:.4f} ({delta:+.4f}) {direction}")

    # Top candidate change
    top_before = results_before[0]["token"]
    top_after = results_after[0]["token"]
    if top_before != top_after:
        print(f"  top-1 changed: {top_before!r} → {top_after!r}")
    else:
        print(f"  top-1 unchanged: {top_before!r}")

    print(f"\n[rl-min] validation complete")

if __name__ == "__main__":
    main()
