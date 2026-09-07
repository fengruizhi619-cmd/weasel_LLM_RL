#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cli_emojiless_exp_v0.2 - in-process llama.cpp candidate tree with
incremental KV cache and branch snapshots via save_state/load_state."""

import argparse, math, os, sys, time, unicodedata
import numpy as np
from llama_cpp import Llama

DEFAULT_MODEL = r"E:\llama.cpp\models\Qwen3-0.6B-Base-Q8_0.gguf"
DEFAULT_WIDTH = 5
DEFAULT_DEPTH = 5
DEFAULT_CTX_CHARS = 100


def is_punct(t):
    return any(unicodedata.category(c).startswith("P") for c in t)

def is_eos_like(t):
    lo = t.lower()
    return "endoftext" in lo or "eos" in lo or "<|im_end|>" in lo or t in ("\n", "\r\n")


class Node:
    __slots__ = ("tok","p","cum","parent","children","is_leaf","stop","depth")
    def __init__(self, tok="", p=1.0, cum=1.0, parent=None, depth=0):
        self.tok = tok; self.p = p; self.cum = cum
        self.parent = parent; self.children = []
        self.is_leaf = False; self.stop = ""; self.depth = depth
    @property
    def path(self):
        parts = []; n = self
        while n and n.parent:
            parts.append(n.tok); n = n.parent
        return "".join(reversed(parts))
    @property
    def is_root(self):
        return self.parent is None


def get_top_candidates(llm, k):
    """Read eval_logits (populated after llm.eval()), return top-k candidates."""
    raw = llm.eval_logits
    logits = np.asarray(list(raw)[-1] if hasattr(raw, "__iter__") and not isinstance(raw, np.ndarray) else raw, dtype=np.float64)
    logits = logits - logits.max()
    exp = np.exp(logits)
    probs = exp / exp.sum()
    top_idx = np.argsort(probs)[::-1][:k]
    results = []
    for idx in top_idx:
        tok_bytes = llm.detokenize([int(idx)])
        tok = tok_bytes.decode("utf-8", errors="replace")
        results.append({"tok": tok, "p": float(probs[idx]), "id": int(idx)})
    return results


def build_tree_dfs(llm, prompt_tokens, width, depth, stats):
    root = Node(depth=0)

    def dfs(node, cur_depth):
        if cur_depth >= depth:
            node.is_leaf = True; node.stop = "depth"
            stats["leaves"] += 1
            return

        candidates = get_top_candidates(llm, width)
        stats["requests"] += 1

        snapshot = llm.save_state()

        for i, cand in enumerate(candidates):
            tok, p, tid = cand["tok"], cand["p"], cand["id"]
            if not tok:
                continue
            cum = node.cum * p
            child = Node(tok=tok, p=p, cum=cum, parent=node, depth=cur_depth + 1)
            node.children.append(child)
            stats["nodes"] += 1

            if is_eos_like(tok):
                child.is_leaf = True; child.stop = "eos"; stats["leaves"] += 1
            elif is_punct(tok):
                child.is_leaf = True; child.stop = "punct"; stats["leaves"] += 1
            else:
                llm.eval([tid])
                dfs(child, cur_depth + 1)
                llm.load_state(snapshot)

        del snapshot

    dfs(root, 0)
    return root, stats


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("-d", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--text", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ctx-chars", type=int, default=DEFAULT_CTX_CHARS)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    text = args.text
    if not text:
        print("[tree] enter prompt:")
        text = "".join(sys.stdin.readlines())
    if not text.strip():
        return 1
    if len(text) > args.ctx_chars:
        text = text[-args.ctx_chars:]

    print(f"[tree] loading model: {args.model}", flush=True)
    llm = Llama(model_path=args.model, n_ctx=1024, n_gpu_layers=-1,
                logits_all=True, verbose=False)
    print(f"[tree] loaded, n_vocab={llm.n_vocab()} n_ctx={llm.n_ctx()}", flush=True)
    print(f"[tree] prompt: {text!r}", flush=True)

    prompt_tokens = llm.tokenize(text.encode("utf-8"))
    print(f"[tree] prompt tokens: {len(prompt_tokens)}", flush=True)

    t0 = time.monotonic()
    llm.eval(prompt_tokens)
    print(f"[tree] prompt eval: {time.monotonic()-t0:.2f}s", flush=True)

    print(f"[tree] building tree width={args.n} depth={args.d} (DFS+incremental KV)...", flush=True)
    t0 = time.monotonic()
    stats = {"requests": 0, "nodes": 1, "leaves": 0}
    root, stats = build_tree_dfs(llm, prompt_tokens, args.n, args.d, stats)
    elapsed = time.monotonic() - t0

    print(f"\n[tree] done in {elapsed:.2f}s | "
          f"requests={stats['requests']} nodes={stats['nodes']} "
          f"leaves={stats['leaves']}", flush=True)

    if args.verbose:
        print("\n=== TREE ===")
        print_tree(root)
    print_leaves(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
