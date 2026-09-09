#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cli_emojiless_exp_v0.2 - complete end-to-end RL loop.

Context reader → build tree → save predictions
→ user types → next commit → compare actual vs predicted
→ reward = p × match_ratio → update lm_head → repeat
"""

import argparse, json, math, os, re, sys, time, unicodedata
import numpy as np
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_MODEL = (os.environ.get("WEASEL_LLM_GGUF", "").strip()
                or os.path.join(_REPO_ROOT, "models", "Qwen3-0.6B-Base-Q8_0.gguf"))
# HF directory (safetensors + tokenizer) for the PyTorch RL side; the GGUF
# above is used by llama.cpp for tree building. They must be passed separately:
# transformers cannot from_pretrained() a bare .gguf path.
DEFAULT_HF_MODEL = (os.environ.get("WEASEL_LLM_MODEL", "").strip()
              or os.path.join(_REPO_ROOT, "models", "Qwen3-0.6B-Base"))
WIDTH = 5
DEPTH = 5
TOP_N = 10
CTX_CHARS = 100
LR = 1e-4
# Tree building goes through llama-server over HTTP (prompt cache + parallel
# slots). The in-process llama_cpp DFS took ~86s per tree; the HTTP path is the
# one that was verified fast before b0a66de.
DEFAULT_ENDPOINT = "http://127.0.0.1:8081/completion"
REQ_TIMEOUT = 20


def is_punct(t):
    return any(unicodedata.category(c).startswith("P") for c in t)

def is_eos_like(t):
    lo = t.lower()
    return "endoftext" in lo or "eos" in lo or "<|im_end|>" in lo or t in ("\n", "\r\n")


class Node:
    __slots__ = ("tok","p","cum","parent","children","is_leaf","stop","depth")
    def __init__(self, tok="", p=1.0, cum=1.0, parent=None, depth=0):
        self.tok=tok; self.p=p; self.cum=cum
        self.parent=parent; self.children=[]
        self.is_leaf=False; self.stop=""; self.depth=depth
    @property
    def path(self):
        parts=[]; n=self
        while n and n.parent: parts.append(n.tok); n=n.parent
        return "".join(reversed(parts))
    @property
    def is_root(self): return self.parent is None


def get_top_candidates(llm, k):
    raw = llm.eval_logits
    logits = np.asarray(list(raw)[-1] if hasattr(raw, "__iter__") and not isinstance(raw, np.ndarray) else raw, dtype=np.float64)
    logits = logits - logits.max()
    exp = np.exp(logits)
    probs = exp / exp.sum()
    top_idx = np.argsort(probs)[::-1][:k]
    out = []
    for idx in top_idx:
        tok_bytes = llm.detokenize([int(idx)])
        tok = tok_bytes.decode("utf-8", errors="replace")
        out.append({"tok": tok, "p": float(probs[idx]), "id": int(idx)})
    return out


def build_tree_dfs(llm, prompt_tokens, width, depth, stats):
    root = Node(depth=0)

    def dfs(node, cur_depth):
        if cur_depth >= depth:
            node.is_leaf = True; node.stop = "depth"
            stats["leaves"] += 1; return
        candidates = get_top_candidates(llm, width)
        stats["requests"] += 1
        branch_ntokens = llm.n_tokens
        snapshot = llm.save_state()
        for i, cand in enumerate(candidates):
            tok, p, tid = cand["tok"], cand["p"], cand["id"]
            if not tok: continue
            cum = node.cum * p
            child = Node(tok=tok, p=p, cum=cum, parent=node, depth=cur_depth + 1)
            node.children.append(child); stats["nodes"] += 1
            if is_eos_like(tok):
                child.is_leaf = True; child.stop = "eos"; stats["leaves"] += 1
            elif is_punct(tok):
                child.is_leaf = True; child.stop = "punct"; stats["leaves"] += 1
            else:
                llm.n_tokens = branch_ntokens
                llm._ctx.kv_cache_seq_rm(-1, branch_ntokens, -1)
                llm.eval([tid])
                dfs(child, cur_depth + 1)
                llm.n_tokens = branch_ntokens
                llm._ctx.kv_cache_seq_rm(-1, branch_ntokens, -1)
        del snapshot

    dfs(root, 0)
    return root, stats


class HttpTree:
    """Candidate lookup against a running llama-server."""

    def __init__(self, endpoint, timeout=REQ_TIMEOUT):
        self.endpoint = endpoint
        self.timeout = timeout

    def top_n(self, prompt, n):
        payload = {
            "prompt": prompt, "n_predict": 1, "n_probs": n,
            "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
            "cache_prompt": True, "post_sampling_probs": True,
        }
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        probs = r.json().get("completion_probabilities", [{}])[0].get(
            "top_probs", [])
        out = []
        for item in probs:
            tok = item.get("token", "")
            if not tok:
                continue
            out.append({"tok": tok, "p": float(item.get("prob", 0.0)),
                        "id": int(item.get("id", 0))})
        return out


def build_tree_http(server, prompt_text, width, depth, stats):
    root = Node(depth=0)

    def dfs(node, full_prompt, cur_depth):
        if cur_depth >= depth:
            node.is_leaf = True
            node.stop = "depth"
            stats["leaves"] += 1
            return
        candidates = server.top_n(full_prompt, width)
        stats["requests"] += 1
        for cand in candidates:
            tok, p = cand["tok"], cand["p"]
            if not tok or ord(tok[0]) < 0x20:
                continue
            child = Node(tok=tok, p=p, cum=node.cum * p, parent=node,
                         depth=cur_depth + 1)
            node.children.append(child)
            stats["nodes"] += 1
            if is_eos_like(tok):
                child.is_leaf = True
                child.stop = "eos"
                stats["leaves"] += 1
            elif is_punct(tok):
                child.is_leaf = True
                child.stop = "punct"
                stats["leaves"] += 1
            else:
                dfs(child, full_prompt + tok, cur_depth + 1)

    dfs(root, prompt_text, 0)
    return root, stats


def collect_leaves(root):
    leaves = []
    def col(n):
        if n.is_leaf and not n.is_root: leaves.append(n)
        for c in n.children: col(c)
    col(root)
    leaves.sort(key=lambda n: n.cum, reverse=True)
    return leaves


def print_tree(node, prefix="", is_last=True):
    if node.is_root:
        print(prefix + "(root)")
    else:
        conn = "\u2514\u2500 " if is_last else "\u251c\u2500 "
        p = f" p={node.p:.4f}"
        stop = f" [{node.stop}]" if node.stop else ""
        print(prefix + conn + repr(node.tok) + p + stop)
    cp = prefix + ("   " if is_last else "\u2502  ")
    for i, c in enumerate(node.children):
        print_tree(c, cp, i == len(node.children) - 1)


def main():
    ap = argparse.ArgumentParser(description="cli_emojiless_exp_v0.2 end-to-end RL")
    ap.add_argument("--log-file", required=True)
    ap.add_argument("-n", type=int, default=WIDTH)
    ap.add_argument("-d", type=int, default=DEPTH)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--model", default=DEFAULT_HF_MODEL)
    ap.add_argument("--llm-model", default=DEFAULT_MODEL)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--ctx-chars", type=int, default=CTX_CHARS)
    ap.add_argument("--rl-lr", type=float, default=LR)
    ap.add_argument("--ckpt-dir", default="")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log_file = os.path.abspath(args.log_file)
    if not os.path.exists(log_file):
        print(f"[v0.2] log not found: {log_file}", file=sys.stderr); return 1
    last_pos = os.path.getsize(log_file)

    # Load PyTorch model for RL
    print(f"[v0.2] loading RL model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to("cuda")
    if model.lm_head.weight is model.model.embed_tokens.weight:
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.data.clone())
    for p_ in model.parameters(): p_.requires_grad = False
    model.lm_head.weight.requires_grad = True
    optimizer = torch.optim.SGD([model.lm_head.weight], lr=args.rl_lr)
    print(f"[v0.2] model ready (lm_head trainable, lr={args.rl_lr})", flush=True)

    # Tree building goes through the running llama-server.
    print(f"[v0.2] tree endpoint: {args.endpoint}", flush=True)
    http_tree = HttpTree(args.endpoint)

    # State for end-to-end RL loop
    prev_tree = None       # last tree's root node
    prev_context = ""      # context used to build last tree
    prev_input_ids = None  # input_ids used for last tree build
    prev_leaves = []       # sorted leaves from last tree

    running = True
    commit_count = 0
    rl_update_count = 0
    import signal as sig_mod
    def handler(sig, frame):
        nonlocal running; running = False
    sig_mod.signal(sig_mod.SIGINT, handler)

    ckpt_dir = os.path.join(os.path.dirname(log_file), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    _dirty = False
    _total_updates = 0

    def save_ckpt():
        nonlocal _dirty
        if not _dirty: return
        w = model.lm_head.weight.data.cpu().half()
        torch.save({"lm_head_weight": w, "updates": _total_updates,
                    "timestamp": time.strftime("%Y%m%d_%H%M%S")},
                   os.path.join(ckpt_dir, "lm_head.pt"))
        _dirty = False
        print(f"[v0.2] [ckpt] saved (updates={_total_updates})", flush=True)

    print(f"\n[v0.2] ready. Type Chinese in any app with Weasel. Ctrl+C to stop.\n", flush=True)

    try:
        while running:
            time.sleep(0.15)
            if not os.path.exists(log_file): continue
            cur_size = os.path.getsize(log_file)
            if cur_size <= last_pos: continue

            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos); new_content = f.read()
            last_pos = os.path.getsize(log_file)

            for line in new_content.splitlines():
                m = re.search(r"ctx\(\d+/\d+\):\s(.+)", line)
                if not m: continue
                ctx_text = m.group(1).strip()
                if not ctx_text: continue
                if len(ctx_text) > args.ctx_chars:
                    ctx_text = ctx_text[-args.ctx_chars:]
                commit_count += 1

                # ==== STEP 1: Check reward against previous tree predictions ====
                reward = 0.0
                reward_info = ""
                if prev_tree and prev_context:
                    if ctx_text.startswith(prev_context):
                        # Characters the user actually typed after the context the
                        # previous tree was built from.
                        new_chars = ctx_text[len(prev_context):]
                        best = None

                        def find_match(node):
                            nonlocal best
                            path = node.path
                            if node.is_leaf and path:
                                n = 0
                                while (n < len(path) and n < len(new_chars)
                                       and path[n] == new_chars[n]):
                                    n += 1
                                if n > 0:
                                    score = node.cum * (n / len(path))
                                    if best is None or score > best[0]:
                                        best = (score, path, n)
                            for c in node.children:
                                find_match(c)

                        find_match(prev_tree)
                        if best:
                            reward = best[0]
                            reward_info = (f"predicted={best[1]!r} "
                                           f"matched={best[2]}/{len(best[1])} "
                                           f"P={reward:.6f}")
                        else:
                            reward_info = "off-path"
                    else:
                        reward_info = "diverged"

                # ==== STEP 2: RL update if we have a positive reward ====
                if reward > 0 and prev_input_ids is not None:
                    # Get hidden state for previous context
                    with torch.no_grad():
                        prev_ids = tokenizer.encode(prev_context, return_tensors="pt").to("cuda")
                        hidden = model.model(input_ids=prev_ids).last_hidden_state[0, -1, :]
                        if hidden.dtype != torch.float32:
                            hidden = hidden.float()

                    # Find target token: the new characters the user typed
                    new_part = ctx_text[len(prev_context):] if ctx_text.startswith(prev_context) else ctx_text
                    target_tokens = tokenizer.encode(new_part, add_special_tokens=False)
                    if target_tokens:
                        target_id = target_tokens[0]  # first new token

                        model.train()
                        logits = model.lm_head(hidden)
                        log_probs = F.log_softmax(logits, dim=-1)
                        loss = -reward * log_probs[target_id]
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        model.eval()
                        _dirty = True
                        _total_updates += 1
                        rl_update_count += 1
                        reward_info += f" [RL] loss={loss.item():.4f}"

                # ==== STEP 3: Build new tree from current context ====
                t0 = time.monotonic()
                root, stats = build_tree_http(http_tree, ctx_text, args.n,
                                              args.d,
                                              {"requests": 0, "nodes": 1,
                                               "leaves": 0})
                elapsed = time.monotonic() - t0
                leaves = collect_leaves(root)

                # Save for next round
                prev_tree = root
                prev_leaves = leaves
                prev_context = ctx_text
                prev_input_ids = tokenizer.encode(ctx_text)

                # Display
                print(f"\n{'='*60}", flush=True)
                print(f"[commit #{commit_count}] ctx={ctx_text!r}", flush=True)
                if reward_info:
                    print(f"[reward] {reward_info}", flush=True)
                if reward > 0:
                    print(f"[RL] reward={reward:.4f} updates={_total_updates}", flush=True)
                print(f"[tree] {elapsed:.2f}s req={stats['requests']} "
                      f"nodes={stats['nodes']} leaves={stats['leaves']}", flush=True)
                if leaves:
                    print(f"[top {min(3, len(leaves))}]:", flush=True)
                    for i, lf in enumerate(leaves[:3], 1):
                        print(f"  {i}. P={lf.cum:.6f} {lf.path!r}", flush=True)

                if _dirty and commit_count % 20 == 0:
                    save_ckpt()

    except Exception as e:
        print(f"[v0.2] [ERROR] {e}", file=sys.stderr)
    finally:
        save_ckpt()
        print(f"\n[v0.2] stopped. commits={commit_count} rl_updates={rl_update_count}", flush=True)
    return 0


def collect_leaves(root):
    leaves = []
    def col(n):
        if n.is_leaf and not n.is_root: leaves.append(n)
        for c in n.children: col(c)
    col(root)
    leaves.sort(key=lambda n: n.cum, reverse=True)
    return leaves


def print_tree(node, prefix="", is_last=True):
    if node.is_root:
        print(prefix + "(root)")
    else:
        conn = "\u2514\u2500 " if is_last else "\u251c\u2500 "
        p = f" p={node.p:.4f}"
        stop = f" [{node.stop}]" if node.stop else ""
        print(prefix + conn + repr(node.tok) + p + stop)
    cp = prefix + ("   " if is_last else "\u2502  ")
    for i, c in enumerate(node.children):
        print_tree(c, cp, i == len(node.children) - 1)


def print_leaves(root):
    leaves = []
    def col(n):
        if n.is_leaf and not n.is_root: leaves.append(n)
        for c in n.children: col(c)
    col(root)
    leaves.sort(key=lambda n: n.cum, reverse=True)
    print(f"\n=== leaves ({len(leaves)}) by cum prob ===")
    for i, lf in enumerate(leaves[:20], 1):
        print(f"  {i:3d}. P={lf.cum:.6f} d={lf.depth} stop={lf.stop:6s} {lf.path!r}")


if __name__ == "__main__":
    sys.exit(main())
