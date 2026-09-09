#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline training over recorded segments.

Reads diag/segments.jsonl (written by offline_recorder.py), and for every
record does exactly what the online loop does:

  commit    -> build a candidate tree from ctx, reward = cum x match ratio,
               one SGD step on lm_head
  backspace -> unlikelihood step on the removed text (weight 1.0/0.3/0.5 as
               classified by the recorder's ctx/app heuristics)

Checkpoints go to the same five diluted slots, so online and offline training
share one weight lineage.

Usage:
  python offline_train.py                     # train once over all new records
  python offline_train.py --epochs 3 --width 20 --depth 10
  python offline_train.py --dry-run           # just report the rewards
"""
import argparse
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
from corpus import CorpusWriter
import unified_pipeline as up
import unified_watcher as uw


def load_records(path, seen_path):
    writer = CorpusWriter(path)
    records = writer.read_all()
    seen = 0
    if os.path.exists(seen_path):
        try:
            seen = int(io.open(seen_path, encoding="utf-8").read().strip() or 0)
        except Exception:
            seen = 0
    return records, records[seen:], seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default=os.path.join(HERE, "diag", "segments.jsonl"))
    ap.add_argument("--seen", default=os.path.join(HERE, "diag", "segments.seen"))
    ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "diag", "checkpoints_online"))
    ap.add_argument("--model", default=up.MODEL_PATH)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default=up.DEVICE)
    ap.add_argument("-n", "--width", type=int, default=20)
    ap.add_argument("-d", "--depth", type=int, default=10)
    ap.add_argument("--rl-lr", type=float, default=up.LR)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records, todo, seen = load_records(args.segments, args.seen)
    print("[offline] %d records total, %d new" % (len(records), len(todo)), flush=True)
    if not todo:
        return 0

    engine = up.TreeEngine(args.model, lr=args.rl_lr, device=args.device,
                           dtype=args.dtype)
    ckpt = uw.CheckpointManager(engine, args.ckpt_dir)
    if ckpt.load_latest():
        print("[offline] resumed updates=%d" % ckpt.updates, flush=True)

    steps = hits = 0
    total = len(todo)
    for epoch in range(max(1, args.epochs)):
        for index, rec in enumerate(todo):
            if (index + 1) % 5 == 0 or index + 1 == total:
                print("[progress] %d/%d hits=%d steps=%d"
                      % (index + 1, total, hits, steps), flush=True)
            ctx = rec.get("ctx", "")
            segment = rec.get("segment", "")
            kind = rec.get("kind", "commit")
            if not segment:
                continue
            if kind == "commit":
                tree, leaves, stats = engine.build_tree(ctx, args.width, args.depth)
                reward, path = uw.find_best_reward(tree, segment)
                if reward <= 0:
                    continue
                hits += 1
                if args.dry_run:
                    continue
                loss = engine.rl_update(ctx, path, reward, args.grad_clip)
            else:
                if args.dry_run:
                    continue
                loss = engine.rl_update_unlikelihood(
                    ctx, segment, rec.get("weight", 0.5), args.grad_clip)
            if loss is None:
                continue
            steps += 1
            ckpt.mark_dirty()
            if steps % args.save_every == 0:
                ckpt.save_epoch(force=True)
                print("[offline] epoch=%d steps=%d hits=%d loss=%.4f updates=%d"
                      % (epoch + 1, steps, hits, loss, ckpt.updates), flush=True)

    if not args.dry_run:
        ckpt.save_epoch(force=True)
    with io.open(args.seen, "w", encoding="utf-8") as f:
        f.write(str(len(records)))
    print("[offline] done: steps=%d hits=%d updates=%d" % (steps, hits, ckpt.updates),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
