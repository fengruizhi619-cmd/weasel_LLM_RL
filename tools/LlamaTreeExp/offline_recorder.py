#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline recorder: context + committed segments. No model, no inference.

For every committed text change it stores the context *before* the change and
the segment the user committed, e.g. "你吃饭了吗" is recorded as
  ctx=""       segment="你"
  ctx="你"     segment="吃饭"
  ctx="你吃饭" segment="了"
  ctx="你吃饭了" segment="吗"
Backspaces are stored as kind="backspace" with the removed text, so the
offline training step can reuse the exact same reward / unlikelihood method.

Usage: pythonw offline_recorder.py [--log-file ...] [--out ...]
"""
import argparse
import io
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from corpus import CorpusWriter

CTX_RE = re.compile(r"ctx\(\d+/\d+\):\s(.+?)\s*$")
FOCUS_RE = re.compile(r"\[focus\] subscribed text-changed on (.+)$")


def common_prefix_len(left, right):
    n = 0
    while n < len(left) and n < len(right) and left[n] == right[n]:
        n += 1
    return n


class Recorder:
    def __init__(self, args):
        self.args = args
        self.writer = CorpusWriter(args.out)
        self.stats_writer = CorpusWriter(args.stats)
        self.prev = None
        self.app = ""
        self.segments = 0
        self.backspaces = 0

    def on_focus(self, line):
        m = FOCUS_RE.search(line)
        if m:
            self.app = m.group(1).strip()
        self.prev = None

    def on_context(self, ctx):
        if len(ctx) > self.args.ctx_chars:
            ctx = ctx[-self.args.ctx_chars:]
        if self.prev is None:
            self.prev = ctx
            return
        if ctx == self.prev:
            return
        common = common_prefix_len(self.prev, ctx)
        if common == len(self.prev):
            kind, segment = "commit", ctx[common:]
        elif common == len(ctx):
            kind, segment = "backspace", self.prev[common:]
        else:
            kind, segment = "replace", self.prev[common:]
        if segment:
            self.writer.write({"t": time.time(), "app": self.app,
                               "ctx": self.prev if kind != "commit" else self.prev,
                               "segment": segment, "kind": kind,
                               "ctx_len": len(self.prev)})
            if kind == "commit":
                self.segments += 1
            else:
                self.backspaces += 1
            if (self.segments + self.backspaces) % 25 == 0:
                self.stats_writer.write({"t": time.time(),
                                         "segments": self.segments,
                                         "backspaces": self.backspaces})
        self.prev = ctx

    def run(self):
        log_file = os.path.abspath(self.args.log_file)
        if not os.path.exists(log_file):
            open(log_file, "a", encoding="utf-8").close()
        pos = os.path.getsize(log_file)
        while True:
            time.sleep(0.2)
            try:
                size = os.path.getsize(log_file)
            except OSError:
                continue
            if size < pos:
                # the hook restarted / the log was rotated
                pos = 0
            if size <= pos:
                continue
            with io.open(log_file, "r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
            pos = os.path.getsize(log_file)
            for line in chunk.splitlines():
                if "[focus]" in line:
                    self.on_focus(line)
                    continue
                m = CTX_RE.search(line)
                if m:
                    try:
                        self.on_context(m.group(1).strip())
                    except Exception as exc:
                        print("[recorder] %r" % (exc,), flush=True)


LOCK_FILE = os.path.join(HERE, "diag", "recorder.lock")


def already_running():
    """Single instance: the recorder must survive any input method, so several
    starters (watchdog, WeaselServer, ghost_mode.py) may race."""
    try:
        with io.open(LOCK_FILE, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except Exception:
        return False
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process -Id %d -ErrorAction SilentlyContinue | ForEach-Object { $_.Id }" % pid],
        capture_output=True, text=True)
    return str(pid) in out.stdout


def write_lock():
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    with io.open(LOCK_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", default=os.path.join(HERE, "diag", "exp-run-v02.log"))
    ap.add_argument("--out", default=os.path.join(HERE, "diag", "segments.jsonl"))
    ap.add_argument("--stats", default=os.path.join(HERE, "diag", "segments_stats.jsonl"))
    ap.add_argument("--ctx-chars", type=int, default=256)
    args = ap.parse_args()
    if already_running():
        print("[recorder] another recorder is already running, exiting", flush=True)
        return 0
    write_lock()
    print("[recorder] background collection: context + segments (any IME)",
          flush=True)
    try:
        Recorder(args).run()
    finally:
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    main()
