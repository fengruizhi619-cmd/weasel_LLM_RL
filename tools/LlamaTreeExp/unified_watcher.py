# -*- coding: utf-8 -*-
"""cli_emojiless_exp_v0.2 - route A unified full-chain watcher.

WeaselExpContextV0 log -> reward check on previous tree
-> unified PyTorch RL update (lm_head) -> rebuild candidate tree
-> multi-slot checkpoint persistence.
"""
import argparse
import torch
import os
import re
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import unified_pipeline as up

CTX_RE = re.compile(r"ctx\(\d+/\d+\):\s(.+?)\s*$")

SLOTS = [
    ("lm_head_t0.pt", 10),
    ("lm_head_t5m.pt", 300),
    ("lm_head_t25m.pt", 1500),
    ("lm_head_t2h.pt", 7200),
    ("lm_head_t12h.pt", 43200),
]


def find_best_reward(root, typed_text):
    if root is None or not typed_text:
        return 0.0, ""
    best = 0.0
    best_path = ""
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            path = child.path
            if not path:
                continue
            if typed_text.startswith(path):
                ratio = 1.0
            elif path.startswith(typed_text):
                # [TRAIN-019] The tree guessed further ahead than the user
                # actually typed. That is a partial hit and used to score 0 -
                # the old code only reached its partial-credit branch when the
                # typed text was LONGER than the path, which cannot happen
                # inside a startswith() test, so the branch was dead.
                ratio = float(len(typed_text)) / float(len(path))
            else:
                continue
            score = child.cum * ratio
            if score > best:
                best = score
                best_path = path
            stack.append(child)
    return best, best_path


def rank_reward_path(root, typed_text, scheme="harmonic", max_len=24):
    """[TRAIN-024] Per-token rank reward along the tree.

    Walks |typed_text| down the tree; at every step ranks the node's children by
    cumulative probability and pays the matched child by its rank. Returns
    (tokens, ranks, rewards, total) where total is the sum of the per-token
    rewards - the sequence score the user described: for a-b-c-d it is
    g(rank_b) + g(rank_c) + g(rank_d).
    """
    if root is None or not typed_text:
        return [], [], [], 0.0
    tokens, ranks, rewards = [], [], []
    node = root
    for ch in typed_text[:max_len]:
        kids = [c for c in node.children if c.tok and not c.is_leaf]
        if not kids:
            break
        kids.sort(key=lambda c: c.cum, reverse=True)
        hit = None
        for i, c in enumerate(kids):
            if len(c.tok) == 1 and c.tok == ch:
                hit = (i, c)
                break
        if hit is None:
            break
        i, child = hit
        tokens.append(child.tok)
        ranks.append(i)
        rewards.append(up.rank_reward(i, len(kids), scheme))
        node = child
    return tokens, ranks, rewards, float(sum(rewards))


def top_path(root, max_len=24):
    """Highest cumulative-probability chain in the tree (the model's own guess).

    Used as the negative target when the user typed something the tree did not
    contain: that guess is what the head should be pushed away from.
    """
    text = ""
    node = root
    while node is not None and len(text) < max_len:
        best = None
        for child in node.children:
            if not child.tok:
                continue
            if best is None or child.cum > best.cum:
                best = child
        if best is None:
            break
        text += best.tok
        node = best
    return text


class CheckpointManager:
    def __init__(self, engine, ckpt_dir, ckpt_dtype="float32"):
        self.engine = engine
        self.ckpt_dir = ckpt_dir
        self.ckpt_dtype = ckpt_dtype
        self.dirty = False
        self.updates = 0
        self.last_save = {}
        # [P1-9] bookkeeping for sync_from_disk()
        self._last_write_ts = 0.0
        self._last_sync_check = 0.0
        os.makedirs(ckpt_dir, exist_ok=True)

    def load_latest(self):
        """P1-6: pick the slot with the highest update count, not the first hit."""
        import torch
        best = None
        for name, _ in SLOTS:
            path = os.path.join(self.ckpt_dir, name)
            if not os.path.exists(path):
                continue
            try:
                meta = torch.load(path, map_location="cpu", mmap=True,
                                  weights_only=False)
                key = (int(meta.get("updates", 0)), str(meta.get("timestamp", "")))
            except Exception:
                continue
            if best is None or key > best[0]:
                best = (key, path)
        if best is None:
            return False
        if not self.engine.load_checkpoint(best[1]):
            return False
        self.updates = best[0][0]
        return True

    def mark_dirty(self):
        self.dirty = True
        self.updates += 1

    def sync_from_disk(self, min_interval_s=15.0):
        """[P1-9] Adopt a newer head written by another process.

        The offline trainer and the online service share these five slots. The
        service keeps its own copy of the head in memory and used to write it
        back on its diluted schedule, which silently rolled back anything the
        trainer had just produced. Before every save we therefore look for a
        slot that is newer than our own last write and carries more updates; if
        one exists we load it instead of overwriting it.

        Returns True when our weights were replaced.
        """
        now = time.time()
        if now - self._last_sync_check < min_interval_s:
            return False
        self._last_sync_check = now

        newest = None
        for name, _interval in SLOTS:
            path = os.path.join(self.ckpt_dir, name)
            try:
                mtime = os.stat(path).st_mtime
            except OSError:
                continue
            if mtime <= self._last_write_ts + 1e-6:
                continue
            if newest is None or mtime > newest[1]:
                newest = (path, mtime)
        if newest is None:
            return False
        try:
            meta = torch.load(newest[0], map_location="cpu", mmap=True,
                              weights_only=False)
            disk_updates = int(meta.get("updates") or 0)
        except Exception:
            return False
        if disk_updates <= self.updates:
            return False
        if not self.engine.load_checkpoint(newest[0]):
            return False
        self.updates = disk_updates
        self._last_write_ts = newest[1]
        return True

    def save_epoch(self, force=False, lock=None):
        adopted = self.sync_from_disk()
        if not self.dirty:
            return adopted
        now = time.time()
        for name, interval in SLOTS:
            last = self.last_save.get(name, 0.0)
            if force or (now - last) >= interval:
                self.engine.save_checkpoint(
                    os.path.join(self.ckpt_dir, name), updates=self.updates,
                    dtype=self.ckpt_dtype, lock=lock)
                self.last_save[name] = now
                self._last_write_ts = now
        self.dirty = False
        return adopted


def main():
    ap = argparse.ArgumentParser(
        description="cli_emojiless_exp_v0.2 unified full-chain watcher")
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--model", default=up.MODEL_PATH)
    ap.add_argument("--device", default=up.DEVICE)
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                    default="bfloat16" if up.DEVICE == "cuda" else "float32")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--rl-lr", type=float, default=up.LR)
    ap.add_argument("-n", type=int, default=up.WIDTH)
    ap.add_argument("-d", type=int, default=up.DEPTH)
    ap.add_argument("--top-n", type=int, default=up.TOP_N)
    ap.add_argument("--ctx-chars", type=int, default=up.CTX_CHARS)
    ap.add_argument("--ckpt-dir", default="")
    args = ap.parse_args()

    log_file = os.path.abspath(args.log_file)
    ckpt_dir = args.ckpt_dir or os.path.join(
        os.path.dirname(log_file), "checkpoints")

    print("[v0.2] unified full-chain watcher: "
          "context -> reward -> tree -> RL -> checkpoint", flush=True)
    print(f"[v0.2] log={log_file} model={args.model} "
          f"width={args.n} depth={args.d} lr={args.rl_lr}", flush=True)

    engine = up.TreeEngine(args.model, lr=args.rl_lr, device=args.device,
                           dtype=args.dtype, fp8=args.fp8)
    ckpt = CheckpointManager(engine, ckpt_dir)
    if ckpt.load_latest():
        print(f"[v0.2] resumed updates={ckpt.updates}", flush=True)

    if not os.path.exists(log_file):
        open(log_file, "a", encoding="utf-8").close()
    last_pos = os.path.getsize(log_file)

    prev_context = None
    prev_tree = None
    commit_count = 0
    running = True

    def stop_handler(_sig, _frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop_handler)

    print("\n[v0.2] ready. Type Chinese with Weasel in another app.\n",
          flush=True)

    try:
        while running:
            time.sleep(0.1)
            if not os.path.exists(log_file):
                continue
            cur_size = os.path.getsize(log_file)
            if cur_size <= last_pos:
                continue
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                new_content = f.read()
            last_pos = os.path.getsize(log_file)

            for line in new_content.splitlines():
                m = CTX_RE.search(line)
                if not m:
                    continue
                ctx_text = m.group(1).strip()
                if not ctx_text:
                    continue
                if len(ctx_text) > args.ctx_chars:
                    ctx_text = ctx_text[-args.ctx_chars:]
                if ctx_text == prev_context:
                    continue
                commit_count += 1

                reward = 0.0
                reward_path = ""
                typed = ""
                if prev_context and prev_tree:
                    if ctx_text.startswith(prev_context):
                        typed = ctx_text[len(prev_context):]
                        reward, reward_path = find_best_reward(prev_tree, typed)
                    else:
                        print(f"[commit #{commit_count}] diverged, skip reward",
                              flush=True)

                rl_line = ""
                if reward > 0.0 and typed:
                    loss = engine.rl_update(prev_context, typed, reward)
                    ckpt.mark_dirty()
                    ckpt.save_epoch()
                    rl_line = (f" reward={reward:.4f} path={reward_path!r} "
                               f"loss={loss:.6f} updates={ckpt.updates}")

                root, leaves, stats = engine.build_tree(
                    ctx_text, args.n, args.d)

                print(f"\n{'=' * 60}", flush=True)
                print(f"[commit #{commit_count}] ctx={ctx_text!r}", flush=True)
                print(f"[tree] {stats['time']:.2f}s "
                      f"nodes={stats['nodes']} leaves={stats['leaves']} "
                      f"forward_calls={stats.get('forward_calls', 0)}",
                      flush=True)
                if leaves:
                    print(f"[top {min(3, len(leaves))}]:", flush=True)
                    for i, leaf in enumerate(leaves[:3], 1):
                        print(f"  {i}. P={leaf.cum:.6f} {leaf.path!r}",
                              flush=True)
                if rl_line:
                    print(f"[RL]{rl_line}", flush=True)

                prev_context = ctx_text
                prev_tree = root

                ckpt.save_epoch()

    except Exception as exc:
        print(f"[v0.2] [ERROR] {exc}", file=sys.stderr, flush=True)
    finally:
        ckpt.save_epoch(force=True)
        print(f"\n[v0.2] stopped. commits={commit_count} "
              f"rl_updates={ckpt.updates}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())