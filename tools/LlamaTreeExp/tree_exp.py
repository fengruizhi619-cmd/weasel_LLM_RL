#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cli_emojiless_tree_exp - candidate tree via llama.cpp server.

Usage:
  tree_exp.py [-n 5] [-d 5] [--text "前文..."] [--model <gguf>] [--server <exe>]

Starts a llama-server on a free port, sends the prompt (last 100 chars),
grows a candidate tree (width=5, depth=5), prints tree with cumulative
probabilities at leaf nodes. Any Unicode punctuation terminates a branch.
"""

import argparse
import math
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import unicodedata

try:
    import requests
except ImportError:
    print("[ERROR] pip install requests", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------- constants

DEFAULT_LLAMA_SERVER = r"E:\llama.cpp\llama-server.exe"
DEFAULT_MODEL = r"E:\llama.cpp\models\Qwen3-0.6B-Base-Q8_0.gguf"
DEFAULT_WIDTH = 5
DEFAULT_DEPTH = 5
DEFAULT_CTX_CHARS = 100
CTX_SIZE = 1024
PARALLEL_SLOTS = 1
MAX_WORKERS = 8
REQUEST_TIMEOUT = 30
SERVER_STARTUP_TIMEOUT = 30

# ---------------------------------------------------------------- node


class TreeNode:
    __slots__ = ("token", "prob", "cum_prob", "parent", "children",
                 "is_leaf", "stop_reason", "depth")

    def __init__(self, token="", prob=1.0, cum_prob=1.0, parent=None, depth=0):
        self.token = token
        self.prob = prob
        self.cum_prob = cum_prob
        self.parent = parent
        self.children = []
        self.is_leaf = False
        self.stop_reason = ""
        self.depth = depth

    @property
    def path_text(self):
        parts = []
        node = self
        while node and node.parent:
            parts.append(node.token)
            node = node.parent
        return "".join(reversed(parts))

    @property
    def is_root(self):
        return self.parent is None


# ---------------------------------------------------------------- helpers


def is_punct(text):
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P"):
            return True
    return False


def is_eos_like(tok_str):
    low = tok_str.lower()
    return ("endoftext" in low or "eos" in low or "<|im_end|>" in low
            or tok_str in ("\n", "\r\n"))


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- server


class LlamaServer:
    def __init__(self, exe, model, port):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self._proc = None
        self._exe = exe
        self._model = model

    def start(self):
        cmd = [
            self._exe,
            "-m", self._model,
            "--port", str(self.port),
            "--ctx-size", str(CTX_SIZE),
            "--parallel", str(PARALLEL_SLOTS),
            "--cont-batching",
            "--no-warmup",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            try:
                r = requests.get(self.base + "/health", timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "ok":
                        return
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError("llama-server did not become healthy in time")

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def query_top_n(self, prompt, n):
        """POST /completion, return list of {tok_str, p} for top n."""
        _t0 = time.monotonic()
        payload = {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": n,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "cache_prompt": True,
        }
        r = requests.post(self.base + "/completion", json=payload,
                          timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        _lat = time.monotonic() - _t0
        if not hasattr(self, "_latencies"):
            self._latencies = []
        self._latencies.append(_lat)
        data = r.json()
        probs_list = data.get("completion_probabilities", [])
        if not probs_list:
            return []
        top_logprobs = probs_list[0].get("top_logprobs", [])
        result = []
        for item in top_logprobs:
            lp = item.get("logprob", -999.0)
            result.append({
                "tok_str": item.get("token", ""),
                "tok_id": item.get("id", 0),
                "logprob": lp,
                "p": math.exp(lp),
            })
        return result


# ---------------------------------------------------------------- tree


def build_tree(server, prompt_text, width, depth):
    """DFS traversal: maximizes KV cache hits (consecutive queries share prefix)."""
    root = TreeNode(depth=0)
    stats = {"requests": 0, "nodes": 1, "leaves": 0}

    def dfs(node, full_prompt, cur_depth):
        if cur_depth >= depth:
            node.is_leaf = True
            node.stop_reason = "depth"
            stats["leaves"] += 1
            return

        candidates = server.query_top_n(full_prompt, width)
        stats["requests"] += 1

        for cand in candidates:
            tok = cand.get("tok_str", "")
            p = cand.get("p", 0.0)
            if not tok:
                continue

            cum = node.cum_prob * p
            child = TreeNode(token=tok, prob=p, cum_prob=cum,
                             parent=node, depth=cur_depth + 1)
            node.children.append(child)
            stats["nodes"] += 1

            if is_eos_like(tok):
                child.is_leaf = True
                child.stop_reason = "eos"
                stats["leaves"] += 1
            elif is_punct(tok):
                child.is_leaf = True
                child.stop_reason = "punct"
                stats["leaves"] += 1
            else:
                dfs(child, full_prompt + tok, cur_depth + 1)

    dfs(root, prompt_text, 0)
    return root, stats



def print_tree(root, file=sys.stdout):
    _print_tree(root, "", True, file)


def print_leaves(root, file=sys.stdout):
    leaves = []

    def _collect(node):
        if node.is_leaf and not node.is_root:
            leaves.append(node)
        for c in node.children:
            _collect(c)

    _collect(root)
    leaves.sort(key=lambda n: n.cum_prob, reverse=True)
    print(f"\n=== leaf nodes ({len(leaves)}) sorted by cumulative probability ===",
          file=file)
    for i, leaf in enumerate(leaves, 1):
        print(f"  {i:3d}. P={leaf.cum_prob:.6f} depth={leaf.depth} "
              f"stop={leaf.stop_reason:6s} text={leaf.path_text!r}",
              file=file)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description="llama.cpp candidate tree experiment")
    ap.add_argument("-n", type=int, default=DEFAULT_WIDTH, help="tree width")
    ap.add_argument("-d", type=int, default=DEFAULT_DEPTH, help="tree depth")
    ap.add_argument("--text", default="", help="prompt text (or read from stdin)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="GGUF model path")
    ap.add_argument("--server", default=DEFAULT_LLAMA_SERVER, help="llama-server exe")
    ap.add_argument("--ctx-chars", type=int, default=DEFAULT_CTX_CHARS,
                    help="max prompt chars")
    ap.add_argument("--top-n", type=int, default=10,
                    help="show top N leaves by cumulative probability")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show full tree and all leaves")
    args = ap.parse_args()

    text = args.text
    if not text:
        print("[tree-exp] enter prompt text (Ctrl+Z / Enter to finish):")
        lines = sys.stdin.readlines()
        text = "".join(lines)

    if not text.strip():
        print("[tree-exp] empty prompt, exiting")
        return 1

    if len(text) > args.ctx_chars:
        text = text[-args.ctx_chars:]
        print(f"[tree-exp] prompt truncated to last {args.ctx_chars} chars")

    port = find_free_port()
    print(f"[tree-exp] starting llama-server on port {port}", flush=True)
    print(f"[tree-exp] model: {args.model}")

    srv = LlamaServer(args.server, args.model, port)
    try:
        srv.start()
        print(f"[tree-exp] server ready", flush=True)

        print(f"[tree-exp] building tree width={args.n} depth={args.d}", flush=True)
        print(f"[tree-exp] prompt: {text!r}", flush=True)

        # Phase 1: warm up - cache base context KV states
        print("[tree-exp] warming up KV cache for base context...", flush=True)
        srv.query_top_n(text, n=1)  # result discarded, just caches KV
        print("[tree-exp] KV cache warmed", flush=True)

        t0 = time.monotonic()
        root, stats = build_tree(srv, text, args.n, args.d)
        elapsed = time.monotonic() - t0

        print(f"[tree-exp] tree built in {elapsed:.2f}s, "
              f"requests={stats['requests']} nodes={stats['nodes']} "
              f"leaves={stats['leaves']}", flush=True)

        # timing breakdown
        if hasattr(srv, "_latencies") and srv._latencies:
            lats = sorted(srv._latencies)
            n_req = len(lats)
            total_req_time = sum(lats)
            print(f"\n[tree-exp] === TIMING BREAKDOWN ===", flush=True)
            print(f"  total requests:     {n_req}", flush=True)
            print(f"  total request time: {total_req_time:.2f}s", flush=True)
            print(f"  avg latency:        {total_req_time/n_req*1000:.1f}ms", flush=True)
            print(f"  min latency:        {lats[0]*1000:.1f}ms", flush=True)
            print(f"  p50:                {lats[n_req//2]*1000:.1f}ms", flush=True)
            print(f"  p95:                {lats[int(n_req*0.95)]*1000:.1f}ms", flush=True)
            print(f"  max latency:        {lats[-1]*1000:.1f}ms", flush=True)
            # concurrency speedup estimate
            sequential_estimate = total_req_time  # if done one by one
            actual_time = elapsed
            speedup = sequential_estimate / actual_time if actual_time > 0 else 0
            print(f"  sequential est:     {sequential_estimate:.2f}s", flush=True)
            print(f"  actual (concurrent):{actual_time:.2f}s", flush=True)
            print(f"  concurrency speedup: {speedup:.1f}x", flush=True)

        # collect leaves
        leaves = []
        def _collect(node):
            if node.is_leaf and not node.is_root:
                leaves.append(node)
            for c in node.children:
                _collect(c)
        _collect(root)
        leaves.sort(key=lambda n: n.cum_prob, reverse=True)

        if args.verbose:
            print("\n" + "=" * 60, flush=True)
            print("CANDIDATE TREE", flush=True)
            print("=" * 60, flush=True)
            print_tree(root, file=sys.stdout)
            print_leaves(root, file=sys.stdout)
        else:
            if leaves:
                print(f"[result] top {min(args.top_n, len(leaves))} of {len(leaves)} leaves:",
                      flush=True)
                for i, leaf in enumerate(leaves[:args.top_n], 1):
                    print(f"  {i:2d}. P={leaf.cum_prob:.6f} {leaf.path_text!r}",
                          flush=True)
            else:
                print("[result] (no leaves)", flush=True)

    except Exception as e:
        print(f"[tree-exp] [ERROR] {e}", file=sys.stderr)
        return 1
    finally:
        srv.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
