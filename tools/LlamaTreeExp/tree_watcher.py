#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tree_watcher.py - cli_emojiless_exp_v0.2
Monitors context reader log file for new commits,
builds candidate tree via llama.cpp, displays top results.

Architecture:
  WeaselExpContextV0.exe (C#, context reader)
    → writes context lines to log file
  tree_watcher.py (this script)
    → tails log file, parses context
    → queries llama-server for candidate tree
    → displays top-N results in cmd
"""

import argparse
import math
import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_exp import LlamaServer, TreeNode, build_tree, print_tree, print_leaves, is_punct, is_eos_like

try:
    import requests
except ImportError:
    print("[ERROR] pip install requests", file=sys.stderr)
    sys.exit(1)

# [V02-001 CONFIG]
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_LLAMA_SERVER = (os.environ.get("WEASEL_LLM_SERVER", "").strip()
                       or os.path.join(_REPO_ROOT, "llama.cpp", "llama-server.exe"))
DEFAULT_MODEL = (os.environ.get("WEASEL_LLM_GGUF", "").strip()
                 or os.path.join(_REPO_ROOT, "models", "Qwen3-0.6B-Base-Q8_0.gguf"))
DEFAULT_WIDTH = 5
DEFAULT_DEPTH = 5
DEFAULT_TOP_N = 5
DEFAULT_CTX_CHARS = 100
CTX_SIZE = 1024
PARALLEL_SLOTS = 8
MAX_WORKERS = 8
REQUEST_TIMEOUT = 30
SERVER_STARTUP_TIMEOUT = 30
LOG_POLL_INTERVAL = 0.15  # seconds, just file size check (not UIA polling)

# [V02-002 LOG-PARSER]
CTX_PATTERN = re.compile(r"ctx\(\d+/\d+\):\s(.+)")

def parse_context_line(line):
    """Extract context text from a log line. Returns None if not a context line."""
    m = CTX_PATTERN.search(line)
    return m.group(1) if m else None


# [V02-003 MAIN]
def main():
    ap = argparse.ArgumentParser(description="cli_emojiless_exp_v0.2 tree watcher")
    ap.add_argument("--log-file", required=True, help="C# reader's log file to monitor")
    ap.add_argument("-n", type=int, default=DEFAULT_WIDTH, help="tree width")
    ap.add_argument("-d", type=int, default=DEFAULT_DEPTH, help="tree depth")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="display top N")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="GGUF model path")
    ap.add_argument("--server", default=DEFAULT_LLAMA_SERVER, help="llama-server exe")
    ap.add_argument("--ctx-chars", type=int, default=DEFAULT_CTX_CHARS)
    args = ap.parse_args()

    log_file = os.path.abspath(args.log_file)
    if not os.path.exists(log_file):
        print(f"[v0.2] log file not found: {log_file}", file=sys.stderr)
        return 1

    # Skip existing content (only process new commits)
    last_pos = os.path.getsize(log_file)
    print(f"[v0.2] watching {log_file} (starting at offset {last_pos})", flush=True)

    # Start llama-server (persistent, reused for all commits)
    port = find_free_port()
    print(f"[v0.2] starting llama-server on port {port}", flush=True)
    srv = LlamaServer(args.server, args.model, port)
    try:
        srv.start()
        print(f"[v0.2] llama-server ready, model={args.model}", flush=True)
        print(f"[v0.2] waiting for commits... (Ctrl+C to stop)\n", flush=True)
    except Exception as e:
        print(f"[v0.2] [ERROR] cannot start server: {e}", file=sys.stderr)
        return 1

    # [V02-004 TAIL-LOOP]
    running = True
    commit_count = 0

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    import signal as sig_module
    sig_module.signal(sig_module.SIGINT, signal_handler)

    try:
        while running:
            time.sleep(LOG_POLL_INTERVAL)

            if not os.path.exists(log_file):
                continue

            current_size = os.path.getsize(log_file)
            if current_size <= last_pos:
                continue

            # Read new content
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                new_content = f.read()
            last_pos = os.path.getsize(log_file)

            # Parse context lines
            for line in new_content.splitlines():
                ctx_text = parse_context_line(line)
                if not ctx_text or not ctx_text.strip():
                    continue

                # Truncate to context length
                if len(ctx_text) > args.ctx_chars:
                    ctx_text = ctx_text[-args.ctx_chars:]

                commit_count += 1
                print(f"\n{'='*60}", flush=True)
                print(f"[commit #{commit_count}] context: {ctx_text!r}", flush=True)
                print(f"{'='*60}", flush=True)

                # Build tree
                t0 = time.monotonic()
                try:
                    root, stats = build_tree(srv, ctx_text, args.n, args.d)
                    elapsed = time.monotonic() - t0

                    # Collect leaves
                    leaves = []
                    def _collect(node):
                        if node.is_leaf and not node.is_root:
                            leaves.append(node)
                        for c in node.children:
                            _collect(c)
                    _collect(root)
                    leaves.sort(key=lambda n: n.cum_prob, reverse=True)

                    print(f"[v0.2] tree: {elapsed:.2f}s "
                          f"req={stats['requests']} nodes={stats['nodes']} "
                          f"leaves={stats['leaves']}", flush=True)

                    if leaves:
                        print(f"[v0.2] top {min(args.top_n, len(leaves))} candidates:", flush=True)
                        for i, leaf in enumerate(leaves[:args.top_n], 1):
                            print(f"  {i:2d}. P={leaf.cum_prob:.6f} {leaf.path_text!r}",
                                  flush=True)
                    else:
                        print(f"[v0.2] (no leaf candidates)", flush=True)

                except Exception as e:
                    print(f"[v0.2] [ERROR] tree build failed: {e}", file=sys.stderr)

    except Exception as e:
        print(f"[v0.2] [ERROR] {e}", file=sys.stderr)
    finally:
        srv.stop()
        print(f"\n[v0.2] stopped. total commits processed: {commit_count}", flush=True)

    return 0


def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    sys.exit(main())
