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
    ap.add_argument("--mode", choices=("fast", "tree"), default="fast",
                    help="fast (default): one forward + top-k reward, no tree; "
                         "tree: legacy full-tree reward")
    ap.add_argument("--topk", type=int, default=20,
                    help="candidates compared against the typed character")
    ap.add_argument("--rank-reward", choices=("harmonic", "linear", "exp"),
                    default="harmonic",
                    help="how a token's rank among the reachable candidates is "
                         "turned into reward (see rank_reward)")
    ap.add_argument("--miss-weight", type=float, default=0.3,
                    help="weight of the negative step taken when the typed "
                         "character is not in the top-k")
    ap.add_argument("--include-ctx", action="store_true",
                    help="also train on the recorded context, not only on the "
                         "committed segment")
    ap.add_argument("--prompt-head", type=int, default=8,
                    help="context characters kept as the prompt when "
                         "--include-ctx slides over the whole record")
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
    total_reward = 0.0
    total = len(todo)
    for epoch in range(max(1, args.epochs)):
        for index, rec in enumerate(todo):
            if (index + 1) % 25 == 0 or index + 1 == total:
                print("[progress] %d/%d hits=%d steps=%d reward=%.1f (%.3f/命中)"
                      % (index + 1, total, hits, steps, total_reward,
                         total_reward / hits if hits else 0.0), flush=True)
            ctx = rec.get("ctx", "")
            segment = rec.get("segment", "")
            kind = rec.get("kind", "commit")

            if args.mode == "fast":
                if kind == "commit":
                    if args.include_ctx:
                        # [TRAIN-022] The recorded context is text the user
                        # really typed - slide over it as well instead of only
                        # using it to condition the prediction.
                        text = ctx + segment
                        head = min(args.prompt_head, max(1, len(text) - 1))
                        run_prompt, run_target = text[:head], text[head:]
                    else:
                        run_prompt, run_target = ctx, segment
                    if not run_target:
                        continue
                    n_steps, n_hits, reward_sum = engine.train_sequence(
                        run_prompt, run_target, k=args.topk,
                        grad_clip=args.grad_clip, miss_weight=args.miss_weight,
                        dry_run=args.dry_run, scheme=args.rank_reward)
                    total_reward += reward_sum
                    steps += n_steps
                    hits += n_hits
                    if n_steps and not args.dry_run:
                        ckpt.mark_dirty()
                        if steps % args.save_every == 0:
                            ckpt.save_epoch(force=True)
                            print("[offline] epoch=%d steps=%d hits=%d updates=%d"
                                  % (epoch + 1, steps, hits, ckpt.updates),
                                  flush=True)
                    continue
                if args.dry_run:
                    continue
                loss = engine.rl_update_unlikelihood(
                    ctx, segment, rec.get("weight", 0.5), args.grad_clip)
                if loss is None:
                    continue
                steps += 1
                ckpt.mark_dirty()
                continue

            # ---- tree mode (legacy) ----
            if not segment:
                continue
            if kind == "commit":
                tree, leaves, stats = engine.build_tree(ctx, args.width, args.depth)
                reward, path = uw.find_best_reward(tree, segment)
                if reward > 0:
                    hits += 1
                    if args.dry_run:
                        continue
                    loss = engine.rl_update(ctx, path, reward, args.grad_clip)
                else:
                    # [TRAIN-021] a miss is a signal too: push the model's own
                    # top guess down instead of dropping the whole record
                    wrong = uw.top_path(tree)
                    if not wrong or args.dry_run:
                        continue
                    loss = engine.rl_update_unlikelihood(
                        ctx, wrong, args.miss_weight, args.grad_clip)
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
        # [TRAIN-023] Only a real run may consume the records: a dry run that
        # still advanced the marker would silently throw the data away.
        with io.open(args.seen, "w", encoding="utf-8") as f:
            f.write(str(len(records)))
    print("[offline] done: steps=%d hits=%d reward=%.1f (%.3f/hit) updates=%d"
          % (steps, hits, total_reward,
             total_reward / hits if hits else 0.0, ckpt.updates), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
