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

By default the records are folded back into the text the user actually typed
and walked once (--mode stream), so no character is scored twice.
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


def build_streams(records, head_chars=64, tail_chars=256):
    """Fold the records back into the text the user actually typed.

    Returns a list of (carry, text). `text` is only ever what the user
    committed, so walking it with a growing prefix scores every character
    exactly once; `carry` is up to `head_chars` of the context that came right
    before it and is a prompt, never a target.

    [CTX-003] Three things were wrong with the first version of this:

      * it folded every record, but a backspace/replace record carries the text
        that was REMOVED - gluing that `segment` back on re-inserted deleted
        text (measured: the sample corpus grew from 2460 to 3653 characters);
      * when the chain broke it started the next stream from `ctx + seg`, which
        re-scored the whole context as if it had just been typed;
      * it chained on `stream.endswith(ctx)`, which cannot work: a stream holds
        what was typed this session, while `ctx` is the last N characters of
        the whole document, pre-existing text included.
    """
    streams = []
    index = {}
    for rec in records:
        if rec.get("kind") != "commit":
            continue
        seg = rec.get("segment", "")
        if not seg:
            continue
        ctx = (rec.get("ctx") or "")[-tail_chars:]
        key = rec.get("app", "")
        i = index.get(key)
        if i is not None:
            carry, text = streams[i]
            known = (carry + text)[-tail_chars:]
            if known and ((len(known) <= len(ctx) and ctx.endswith(known))
                          or (len(known) > len(ctx) and known.endswith(ctx))):
                streams[i] = (carry, text + seg)
                continue
        # the document moved under us (edit outside the IME, focus switch):
        # start a fresh stream, carrying only the context in front of it
        streams.append((ctx[-head_chars:], seg))
        index[key] = len(streams) - 1
    return streams


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


def run_streams(args, engine, ckpt, todo, records):
    """[TRAIN-032] Walk every stream once, one character at a time.

    The rule: with "abcd", "a" is the context and only "b" is scored; the next
    step has "ab" as the context and scores only "c". Every character is a
    target exactly once and is never shown to the model twice.

    train_sequence already means exactly that - the prompt grows by one
    character per step and only the new character is rewarded - so a stream
    only has to be walked in chunks. The chunk boundary exists to bound the KV
    cache; the chunk then becomes part of the context for the next one, it is
    never scored again. That is the whole difference from the per-record mode,
    where consecutive records repeat 35..100 characters of context.
    """
    streams = build_streams(todo, args.prompt_head)
    total_chars = sum(len(text) for _carry, text in streams)
    print("[offline] stream mode: %d stream(s) %d chars (prompt-head=%d chunk=%d)"
          % (len(streams), total_chars, args.prompt_head, args.chunk),
          flush=True)
    if not streams:
        return 0

    if args.epochs > 1:
        # [TRAIN-036] the whole point of stream mode is that every character is
        # scored once; a second pass would re-score all of them.
        print("[offline] stream mode ignores --epochs %d (it would re-score the "
              "same characters)" % args.epochs, flush=True)

    steps = hits = 0
    total_loss = 0.0
    since_save = 0
    for epoch in range(1):
        for si, (carry, stream) in enumerate(streams):
            pos = 0
            while pos < len(stream):
                head = (stream[max(0, pos - args.prompt_head):pos]
                        if pos else carry)
                if not head:
                    # [TRAIN-033] "abcd": "a" is the context and only "b" is
                    # scored - the first character of a stream has nothing in
                    # front of it, so it can only ever be context. Without this
                    # the head-less call returns (0, 0, 0) and short streams
                    # would be dropped whole.
                    pos += 1
                    continue
                chunk = stream[pos:pos + args.chunk]
                if not chunk:
                    break
                pos += len(chunk)
                n_steps, n_hits, loss_sum = engine.train_sequence(
                    head, chunk, k=args.topk, grad_clip=args.grad_clip, dry_run=args.dry_run)
                total_loss += loss_sum
                steps += n_steps
                hits += n_hits
                print("[progress] %d/%d pos=%d/%d hits=%d steps=%d loss=%.1f "
                      "(%.4f/步)"
                      % (si + 1, len(streams), pos, len(stream), hits, steps,
                         total_loss, total_loss / steps if steps else 0.0),
                      flush=True)
                if n_steps and not args.dry_run:
                    ckpt.mark_dirty()
                    since_save += n_steps
                    if since_save >= args.save_every:
                        since_save = 0
                        ckpt.save_epoch()
                        print("[offline] epoch=%d steps=%d hits=%d updates=%d"
                              % (epoch + 1, steps, hits, ckpt.updates),
                              flush=True)

    if not args.dry_run:
        # [TRAIN-036] only the realtime slot is stamped with the freshly trained
        # head; forcing all five would overwrite every rollback point.
        ckpt.save_epoch(force=True, slots=("lm_head_t0.pt",))
        with io.open(args.seen, "w", encoding="utf-8") as f:
            f.write(str(len(records)))
    print("[offline] done: steps=%d hits=%d loss=%.1f (%.4f/步) updates=%d"
          % (steps, hits, total_loss,
             total_loss / steps if steps else 0.0, ckpt.updates), flush=True)
    return 0


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
    # [CTX-008] 0 = off (see the note in train_text_corpus.py)
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", choices=("fast", "stream", "tree"),
                    default="stream",
                    help="stream (default): fold the records back into "
                         "continuous text so every character is a target "
                         "exactly once; fast: per-record top-k reward (the "
                         "records overlap, so the same characters get counted "
                         "many times); tree: legacy full-tree reward")
    ap.add_argument("--topk", type=int, default=20,
                    help="candidates compared against the typed character")
    ap.add_argument("--miss-weight", type=float, default=0.3,
                    help="weight of the negative step in --mode tree "
                         "(train_sequence has no miss penalty)")
    ap.add_argument("--prompt-head", type=int, default=64,
                    help="context characters kept in front of every stream "
                         "chunk (a condition, never a target)")
    ap.add_argument("--chunk", type=int, default=128,
                    help="stream characters scored per forward pass; it only "
                         "bounds the KV cache, the chunk itself is context for "
                         "the next one")
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

    if args.mode == "stream":
        return run_streams(args, engine, ckpt, todo, records)

    steps = hits = 0
    total_loss = 0.0
    total = len(todo)
    for epoch in range(max(1, args.epochs)):
        for index, rec in enumerate(todo):
            if (index + 1) % 25 == 0 or index + 1 == total:
                print("[progress] %d/%d hits=%d steps=%d loss=%.1f (%.4f/步)"
                      % (index + 1, total, hits, steps, total_loss,
                         total_loss / steps if steps else 0.0), flush=True)
            ctx = rec.get("ctx", "")
            segment = rec.get("segment", "")
            kind = rec.get("kind", "commit")

            if args.mode == "fast":
                if kind == "commit":
                    # [TRAIN-031] The context is a CONDITION, never a target:
                    # it is the same text the previous records already scored.
                    run_prompt, run_target = ctx, segment
                    if not run_target:
                        continue
                    n_steps, n_hits, loss_sum = engine.train_sequence(
                        run_prompt, run_target, k=args.topk,
                        grad_clip=args.grad_clip,
                        dry_run=args.dry_run)
                    total_loss += loss_sum
                    steps += n_steps
                    hits += n_hits
                    if n_steps and not args.dry_run:
                        ckpt.mark_dirty()
                        if steps % args.save_every == 0:
                            ckpt.save_epoch()
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
                ckpt.save_epoch()
                print("[offline] epoch=%d steps=%d hits=%d loss=%.4f updates=%d"
                      % (epoch + 1, steps, hits, loss, ckpt.updates), flush=True)

    if not args.dry_run:
        ckpt.save_epoch(force=True, slots=("lm_head_t0.pt",))
        # [TRAIN-023] Only a real run may consume the records: a dry run that
        # still advanced the marker would silently throw the data away.
        with io.open(args.seen, "w", encoding="utf-8") as f:
            f.write(str(len(records)))
    print("[offline] done: steps=%d hits=%d loss=%.1f (%.4f/步) updates=%d"
          % (steps, hits, total_loss,
             total_loss / steps if steps else 0.0, ckpt.updates), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
