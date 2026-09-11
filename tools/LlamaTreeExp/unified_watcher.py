# -*- coding: utf-8 -*-
"""cli_emojiless_exp_v0.2 - route A unified full-chain watcher.

WeaselExpContextV0 log -> reward check on previous tree
-> unified PyTorch RL update (lm_head) -> rebuild candidate tree
-> multi-slot checkpoint persistence.
"""
import torch
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


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




class CheckpointManager:
    def __init__(self, engine, ckpt_dir, ckpt_dtype="float32"):
        self.engine = engine
        self.ckpt_dir = ckpt_dir
        self.ckpt_dtype = ckpt_dtype
        self.dirty = False
        self.updates = 0
        # [TRAIN-036] Seed the schedule from the files on disk. These five slots
        # are shared with the offline trainer, and a fresh process used to start
        # with an empty table, so its very first save_epoch() saw every interval
        # as already elapsed and wrote all five slots with the same head - which
        # flattened the dilution ladder that maybe_rollback() depends on.
        self.last_save = {}
        for _name, _interval in SLOTS:
            try:
                self.last_save[_name] = os.path.getmtime(
                    os.path.join(ckpt_dir, _name))
            except OSError:
                pass
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

    def save_epoch(self, force=False, lock=None, slots=None):
        """Write the slots that are due.

        `force` bypasses the interval; `slots` restricts which ones are
        considered, so a caller can insist on the realtime slot without also
        stamping the day-old rollback points with a brand new head.
        """
        adopted = self.sync_from_disk()
        if not self.dirty:
            return adopted
        now = time.time()
        for name, interval in SLOTS:
            if slots is not None and name not in slots:
                continue
            last = self.last_save.get(name, 0.0)
            if force or (now - last) >= interval:
                self.engine.save_checkpoint(
                    os.path.join(self.ckpt_dir, name), updates=self.updates,
                    dtype=self.ckpt_dtype, lock=lock)
                self.last_save[name] = now
                self._last_write_ts = now
        self.dirty = False
        return adopted

