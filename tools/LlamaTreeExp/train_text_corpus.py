#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the head on a plain text corpus - one pass, no repetition.

The live path and the offline segment pass already share one reward rule
(per-token rank reward over a KV-cached walk, see TreeEngine.train_sequence).
This tool pushes arbitrary text through exactly that walk, so an author's own
documents can seed the head long before enough typed records exist.

usage:
  python train_text_corpus.py <file> [--chars N] [--start N] [--end N]
                              [--chunk 400] [--prompt 64]
                              [--ckpt-dir DIR] [--save-every 200] [--dry-run]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import unified_pipeline as up
import unified_watcher as uw


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gbk", "utf-16"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    raise SystemExit("cannot decode " + path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--chars", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "diag", "checkpoints_online"))
    ap.add_argument("--model", default=up.MODEL_PATH)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default=up.DEVICE)
    ap.add_argument("--rl-lr", type=float, default=up.LR)
    ap.add_argument("--topk", type=int, default=20)
    # [CTX-008] 0 = off. A norm clip at 1.0 would shrink every step by 1/68 and,
    # because it normalises the norm, hand a rank-20 candidate the same step as a
    # rank-1 one - the reward would stop deciding the step size.
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text = read_text(args.file)
    if args.start or args.end:
        text = text[args.start:args.end or len(text)]
    if args.chars:
        text = text[:args.chars]
    print("[corpus] %s: %d characters, lr=%g" % (args.file, len(text), args.rl_lr), flush=True)

    engine = up.TreeEngine(args.model, lr=args.rl_lr, device=args.device, dtype=args.dtype)
    ckpt = uw.CheckpointManager(engine, args.ckpt_dir)
    if ckpt.load_latest():
        print("[corpus] resumed updates=%d" % ckpt.updates, flush=True)

    steps = hits = 0
    loss_sum = 0.0
    t0 = time.time()
    last_save = 0
    for i in range(0, len(text), args.chunk):
        # [TRAIN-033] the very first character has nothing in front of it, so it
        # can only be context: shift the window by one. Without this the i=0
        # call had an empty prompt, train_sequence returned (0, 0, 0) and the
        # first whole chunk produced no gradient at all, silently.
        start = i + 1 if i == 0 else i
        prompt = text[max(0, start - args.prompt):start]
        target = text[start:start + args.chunk]
        if not target:
            continue
        s, h, r = engine.train_sequence(
            prompt, target, k=args.topk, grad_clip=args.grad_clip,
            dry_run=args.dry_run)
        steps += s
        hits += h
        loss_sum += r
        if (i // args.chunk) % 5 == 4 or i + args.chunk >= len(text):
            print("[progress] %d/%d chars | steps=%d hits=%d loss=%.1f"
                  % (min(len(text), i + args.chunk), len(text), steps, hits, loss_sum),
                  flush=True)
        if not args.dry_run and steps - last_save >= args.save_every:
            last_save = steps
            ckpt.mark_dirty()
            ckpt.save_epoch()

    if not args.dry_run:
        ckpt.mark_dirty()
        # [TRAIN-036] the realtime slot only - forcing all five would overwrite
        # every rollback point with the head this run just produced.
        ckpt.save_epoch(force=True, slots=("lm_head_t0.pt",))
    print("[corpus] done: steps=%d hits=%d (%.1f%%) loss=%.1f in %.0fs updates=%d"
          % (steps, hits, 100.0 * hits / max(1, steps), loss_sum,
             time.time() - t0, ckpt.updates), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
