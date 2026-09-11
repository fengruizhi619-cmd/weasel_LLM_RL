#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Online unified engine service (S1-S5).

One PyTorch model (frozen fp16 backbone + trainable fp32 lm_head) does two jobs:
  * serve a llama-server compatible /completion API for the IME front end (S1)
  * run the single-sample online RL loop over the context log (S2/S5)

Endpoints
  POST /completion   llama-server compatible top-k probabilities
  GET  /health       liveness
  GET  /metrics      counters (S4)
  POST /feedback     optional explicit accept/reject signal (S4 hook for TSF)

Extra request field
  "pinyin": "mingt"  constrain candidates to that pinyin prefix (S3)
"""
import argparse
import base64
import io
import queue
from collections import OrderedDict
from concurrent.futures import Future
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

import ctxwin
from corpus import CorpusWriter
import unified_pipeline as up
import unified_watcher as uw

CTX_RE = re.compile(r"ctx\(\d+/\d+\):\s(.+?)\s*$")

try:
    from pypinyin import lazy_pinyin
except Exception:  # pragma: no cover
    lazy_pinyin = None

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover
    AESGCM = None


def norm_pinyin(text):
    return re.sub(r"[^a-z]", "", text.lower())


def token_pinyin(tok):
    """Pinyin used by the prefix constraint for one candidate.

    A tree node holds exactly one character, so the constraint has to work on
    the head character's syllable. Using the whole token's pinyin ("dagai" for
    the token <da gai>) consumed the entire typed prefix in a single step and
    left the rest of the path unconstrained - which is why only the first
    character ever looked constrained. Latin/digits still pass through as-is.
    """
    if not tok:
        return ""
    head = tok[0]
    if lazy_pinyin is not None and "\u4e00" <= head <= "\u9fff":
        return norm_pinyin("".join(lazy_pinyin(head)))
    if lazy_pinyin is None:
        return norm_pinyin(tok)
    return norm_pinyin(tok)


def common_prefix_len(left, right):
    n = 0
    while n < len(left) and n < len(right) and left[n] == right[n]:
        n += 1
    return n


def tree_predicts(root, text):
    """True if |text| starts with any predicted path in the tree."""
    if root is None or not text:
        return False
    stack = [root]
    while stack:
        node = stack.pop()
        path = node.path
        if path and text.startswith(path):
            return True
        stack.extend(node.children)
    return False


def classify_backspace(prev_ctx, ctx, trees, deleted=None):
    """P3-1: tell apart the three rejection shapes.

    Returns (kind, deleted, weight):
      predicted-reject  the deleted text was a model prediction  -> weight 1.0
      typing-reject     the user deleted their own typing        -> weight 0.3
      replace           deleted then retyped something else      -> weight 0.5
    """
    if not prev_ctx or not ctx or prev_ctx == ctx:
        return "", "", 0.0
    if deleted is None:
        # [CTX-001] Fallback only: the caller normally hands us the text from
        # ctxwin, because this prefix test cannot see a deletion once the
        # context window has started to slide.
        common = common_prefix_len(prev_ctx, ctx)
        if common == len(ctx):
            deleted = prev_ctx[len(ctx):]
        else:
            return "replace", prev_ctx[common:], 0.5
    if not deleted:
        return "", "", 0.0
    # The text may have been predicted by the tree in force one or two
    # commits ago, so check every recent tree.
    if any(tree_predicts(tree, deleted) for tree in (trees or [])):
        return "predicted-reject", deleted, 1.0
    return "typing-reject", deleted, 0.3


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.start = time.time()
        self.requests = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.latency_ms = []
        self.empty_results = 0
        self.pinyin_filtered = 0
        self.rl_updates = 0
        self.rl_rewards = 0
        self.rl_backspaces = 0
        self.rollbacks = 0
        self.shown = 0
        self.accepts = 0
        self.rejects = 0
        self.commits = 0
        self.backspace_predicted = 0
        self.backspace_typing = 0
        self.backspace_replace = 0
        self.reject_too_long = 0
        self.accept_too_long = 0
        self.batched_requests = 0
        self.batch_saved = 0
        self.accepted_chars = 0
        self.keystrokes = 0
        self.keys_per_char = 3.5
        self.served_probs = []
        self.apps = {}
        self.phases = {"forward": [], "sample": [], "decode": []}

    def bump(self, name, value=1):
        with self.lock:
            setattr(self, name, getattr(self, name) + value)

    def phase(self, name, ms):
        with self.lock:
            values = self.phases.setdefault(name, [])
            values.append(ms)
            if len(values) > 500:
                del values[:-500]

    def hit_rate(self):
        with self.lock:
            return self.rl_rewards / max(1, self.commits)

    def bump_app(self, app, field, value=1):
        if not app:
            return
        with self.lock:
            entry = self.apps.setdefault(
                app, {"requests": 0, "shown": 0, "accepts": 0,
                      "chars": 0, "blocked": 0})
            entry[field] = entry.get(field, 0) + value

    def snapshot(self):
        with self.lock:
            lat = sorted(self.latency_ms)
            def pct(p):
                if not lat:
                    return 0.0
                return lat[min(len(lat) - 1, int(len(lat) * p))]
            shown = max(1, self.shown)
            keys = max(1, self.keystrokes)
            # P2-9: three separate scopes, so the two "rates" are never
            # confused. "training" is IME-independent (driven by the context
            # stream); "ime_feedback" only fills up when the Weasel TSF is the
            # active input method.
            return {
                "service": {
                    "uptime_s": round(time.time() - self.start, 1),
                    "requests": self.requests,
                    "cache_hits": self.cache_hits,
                    "prompt_tokens": self.prompt_tokens,
                    "latency_p50_ms": round(pct(0.5), 1),
                    "latency_p99_ms": round(pct(0.99), 1),
                    "empty_results": self.empty_results,
                    "pinyin_filtered": self.pinyin_filtered,
                    "batched_requests": self.batched_requests,
                    "batch_saved": self.batch_saved,
                    "avg_candidates": round(
                        sum(self.served_probs) / len(self.served_probs), 1)
                    if self.served_probs else 0,
                    "apps": self.apps,
                    "torch_mb": {
                        "allocated": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
                        "reserved": round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
                        "max_reserved": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1),
                    },
                    "phase_p50_ms": {
                        name: (round(sorted(values)[len(values) // 2], 1)
                               if values else 0.0)
                        for name, values in self.phases.items()
                    },
                },
                "training": {
                    "commits": self.commits,
                    "rl_rewards": self.rl_rewards,
                    "rl_updates": self.rl_updates,
                    "rl_backspaces": self.rl_backspaces,
                    "backspace_predicted": self.backspace_predicted,
                    "backspace_typing": self.backspace_typing,
                    "backspace_replace": self.backspace_replace,
                    "reject_too_long": self.reject_too_long,
                    "hit_rate": round(self.rl_rewards / max(1, self.commits), 4),
                    "rollbacks": self.rollbacks,
                },
                "ime_feedback": {
                    "note": "only populated when the Weasel TSF is active",
                    "shown": self.shown,
                    "accepts": self.accepts,
                    "accept_rate": round(self.accepts / shown, 4),
                    "accepted_chars": self.accepted_chars,
                    "keystrokes": self.keystrokes,
                    "savings_rate": round(
                        self.accepted_chars * self.keys_per_char / keys, 4),
                },
            }


class EngineService:
    """Model owner: serves top-k probabilities and performs RL steps."""

    def __init__(self, engine, ckpt, corpus, metrics, args):
        self.engine = engine
        self.ckpt = ckpt
        self.corpus = corpus
        self.metrics = metrics
        self.args = args
        self.lock = threading.RLock()
        self.cache_tokens = []
        self.cache_kv = None
        self.cache_logits = None
        # [P3-4] LRU of (token tuple -> logits). A tree walk revisits the same
        # prefix after every child, so this removes most of the fixed per-token
        # forward cost without touching the model.
        self.logits_cache = OrderedDict()
        self.logits_cache_size = 64
        # [P3-4] token id -> surface cache. Decoding 20 ids per request through
        # the HF tokenizer was the real fixed cost (~30ms).
        self.token_cache = {}
        self.pinyin_cache = {}
        self.last_request = 0.0
        self.last_shown = None
        self.baseline = None
        self.blacklist = {name.strip().lower()
                          for name in (args.blacklist or "").split(",")
                          if name.strip()}
        self.batching_enabled = args.batch_size > 1
        self.batch_queue = queue.Queue()
        self.batch_thread = None
        self.probe = [
            ("天阴沉沉的，明天的天气", "怎么样"),
            ("今天吃饭了吗，我还没", "吃"),
            ("所以我们需要找到一个", "能力"),
        ]

    # ---------- S1: llama-server compatible completion ----------
    def _parse(self, req):
        """Normalise one request. None = blacklisted (answer with nothing)."""
        app = str(req.get("app", "") or "")
        self.metrics.bump_app(app, "requests")
        if app and app.lower() in self.blacklist:
            self.metrics.bump_app(app, "blocked")
            return None
        return {
            "req": req,
            "app": app,
            "ids": self.engine.tokenizer.encode(req.get("prompt", ""),
                                                add_special_tokens=False),
            "n_probs": int(req.get("n_probs", 20) or 20),
            "repeat_penalty": float(req.get("repeat_penalty", 1.0) or 1.0),
            "repeat_last_n": int(req.get("repeat_last_n", 64) or 64),
            "temperature": float(req.get("temperature", 1.0) or 1.0),
            "top_k": int(req.get("top_k", 0) or 0),
            "top_p": float(req.get("top_p", 1.0) or 1.0),
            "min_p": float(req.get("min_p", 0.0) or 0.0),
            "pinyin": norm_pinyin(req.get("pinyin", "") or ""),
        }

    @staticmethod
    def _empty_response():
        return {"content": "", "stop": True, "tokens_predicted": 0,
                "completion_probabilities": [{"token": "", "prob": 0.0,
                                              "id": -1, "top_probs": []}]}

    def complete(self, req, t0=None):
        """Sequential path (also the fallback when batching is off)."""
        parsed = self._parse(req)
        if parsed is None:
            return self._empty_response()
        if t0 is None:
            t0 = time.perf_counter()
        with self.lock:
            self.last_request = time.time()
            t_fwd = time.perf_counter()
            logits, cached = self._forward_cached(parsed["ids"])
            self.metrics.phase("forward", (time.perf_counter() - t_fwd) * 1000)
            self.metrics.bump("requests")
            self.metrics.bump("prompt_tokens", len(parsed["ids"]))
            if cached:
                self.metrics.bump("cache_hits")
            return self._finish(parsed, logits, t0)

    def _finish(self, parsed, logits, t0):
        """Sampling + response. Caller holds self.lock."""
        ids = parsed["ids"]
        n_probs = parsed["n_probs"]
        repeat_penalty = parsed["repeat_penalty"]
        repeat_last_n = parsed["repeat_last_n"]
        temperature = parsed["temperature"]
        top_k = parsed["top_k"]
        top_p = parsed["top_p"]
        min_p = parsed["min_p"]
        pinyin = parsed["pinyin"]
        if True:
            scores = logits.float().clone()
            if repeat_penalty != 1.0 and repeat_last_n > 0:
                for token_id in set(ids[-repeat_last_n:]):
                    if scores[token_id] > 0:
                        scores[token_id] /= repeat_penalty
                    else:
                        scores[token_id] *= repeat_penalty
            if temperature > 0 and temperature != 1.0:
                scores = scores / temperature
            if top_k > 0:
                kth = torch.topk(scores, min(top_k, scores.numel())).values[-1]
                scores[scores < kth] = -float("inf")
            probs = F.softmax(scores, dim=-1)
            if min_p > 0:
                probs[probs < probs.max() * min_p] = 0.0
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
            total = probs.sum()
            if total > 0:
                probs = probs / total

            # S6: dynamic branching budget. When the service is getting slow,
            # return fewer candidates so the front end expands a narrower tree.
            limit = n_probs
            # [S3] A live composition needs the wide probe: the pinyin filter
            # must be able to find a matching syllable inside the sampled set,
            # and the dynamic shrink starves it (measured 200 -> 55 under load).
            if self.args.dynamic and not pinyin and len(self.metrics.latency_ms) >= 20:
                factor = 1.0
                p50 = self._p50_latency()
                if p50 > self.args.latency_budget_ms:
                    factor *= max(0.2, self.args.latency_budget_ms / p50)
                hit = self.metrics.hit_rate()
                if hit >= self.args.hit_rate_high:
                    # the tree is already hitting often -> spend less compute
                    factor *= 0.7
                elif hit < self.args.hit_rate_low:
                    # missing too much -> keep the full width
                    factor = 1.0
                limit = max(self.args.min_probs, int(n_probs * factor))
            t_smp = time.perf_counter()
            count = min(limit, int((probs > 0).sum().item()) or limit)
            top_probs, top_ids = torch.topk(probs, max(1, count))
            self.metrics.phase("sample", (time.perf_counter() - t_smp) * 1000)

            t_dec = time.perf_counter()
            entries = []
            for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
                token = self.token_cache.get(int(token_id))
                if token is None:
                    token = self.engine.tokenizer.decode([int(token_id)])
                    self.token_cache[int(token_id)] = token
                if not token:
                    continue
                py = self.pinyin_cache.get(token)
                if py is None:
                    py = token_pinyin(token)
                    self.pinyin_cache[token] = py
                entries.append({"id": int(token_id), "token": token,
                                "prob": float(prob), "pinyin": py})

            self.metrics.phase("decode", (time.perf_counter() - t_dec) * 1000)
            if pinyin:
                filtered = []
                for item in entries:
                    # [CTX-004] token_pinyin was already run (and cached) when
                    # the entry was built a few lines up - recomputing it here
                    # put pypinyin on the request path for every candidate.
                    tp = item["pinyin"]
                    # [S3] Two-way prefix match: the token may extend what is
                    # typed (tp.startswith) or be a prefix of it (head syllable
                    # "da" while the user is still at "daga"), otherwise a
                    # partially typed syllable drops every candidate.
                    if (tp.startswith(pinyin)
                            or (tp and pinyin.startswith(tp))
                            or (not tp and item["token"].lower().startswith(pinyin))):
                        filtered.append(item)
                if filtered:
                    if len(filtered) != len(entries):
                        self.metrics.bump("pinyin_filtered")
                    entries = filtered

            if not entries:
                self.metrics.bump("empty_results")
            with self.metrics.lock:
                self.metrics.served_probs.append(len(entries))
                if len(self.metrics.served_probs) > 2000:
                    self.metrics.served_probs = self.metrics.served_probs[-2000:]

        top_ids_snapshot = [int(i) for i in top_ids.tolist()] if entries else []
        head = entries[0] if entries else {"token": "", "prob": 0.0, "id": -1}
        self.prefetch_top1(ids, top_ids_snapshot)
        elapsed = (time.perf_counter() - t0) * 1000.0
        with self.metrics.lock:
            self.metrics.latency_ms.append(elapsed)
            if len(self.metrics.latency_ms) > 2000:
                self.metrics.latency_ms = self.metrics.latency_ms[-2000:]
        return {
            "content": head["token"],
            "stop": False,
            "tokens_predicted": 1,
            "completion_probabilities": [{
                "id": head["id"],
                "token": head["token"],
                "prob": head["prob"],
                "top_probs": entries,
            }],
        }

    # ---------- sibling batching (one prefill + one batched step) ----------
    def start_batcher(self):
        if not self.batching_enabled or self.batch_thread is not None:
            return
        self.batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
        self.batch_thread.start()

    def submit(self, req):
        """HTTP entry point: batched when possible, sequential otherwise."""
        if not self.batching_enabled:
            return self.complete(req)
        future = Future()
        self.batch_queue.put((req, future))
        return future.result(timeout=self.args.batch_timeout)

    def _batch_loop(self):
        while True:
            first = self.batch_queue.get()
            batch = [first]
            deadline = time.time() + self.args.batch_wait_ms / 1000.0
            while len(batch) < self.args.batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.batch_queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                results = self._run_batch([item[0] for item in batch])
            except Exception as exc:
                for _req, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (_req, fut), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)

    @staticmethod
    def _lcp(parsed):
        ids_list = [p["ids"] for p in parsed]
        shortest = min(len(x) for x in ids_list)
        lcp = 0
        while lcp < shortest and all(x[lcp] == ids_list[0][lcp] for x in ids_list):
            lcp += 1
        return lcp

    def _run_batch(self, reqs):
        t0 = time.perf_counter()
        if len(reqs) < 2:
            return [self.complete(reqs[0], t0)]
        parsed = [self._parse(r) for r in reqs]
        if any(p is None for p in parsed):
            return [self.complete(r, t0) for r in reqs]
        with self.lock:
            self.last_request = time.time()
            lcp = self._lcp(parsed)
            if lcp > 0 and all(len(p["ids"]) == lcp + 1 for p in parsed):
                return self._finish_shared(parsed, lcp, t0)
        return [self.complete(r, t0) for r in reqs]

    def _finish_shared(self, parsed, lcp, t0):
        """Sibling requests share the prefix: prefill once, step once."""
        t_fwd = time.perf_counter()
        prefix = parsed[0]["ids"][:lcp]
        self._forward_cached(prefix)
        kv = self.cache_kv
        if kv is None or not hasattr(kv, "batch_repeat_interleave"):
            return [self.complete(p["req"], t0) for p in parsed]
        count = len(parsed)
        kv.batch_repeat_interleave(count)
        suffix = torch.tensor([[p["ids"][-1]] for p in parsed],
                              device=self.engine.device)
        with torch.no_grad():
            logits, _ = self.engine.backbone_logits(suffix, kv)
        self.metrics.phase("forward", (time.perf_counter() - t_fwd) * 1000)
        self.metrics.bump("requests", count)
        self.metrics.bump("prompt_tokens", sum(len(p["ids"]) for p in parsed))
        self.metrics.bump("batched_requests", count)
        self.metrics.bump("batch_saved", count - 1)
        self.cache_tokens = []
        self.cache_kv = None
        self.cache_logits = None
        return [self._finish(p, logits[i], t0) for i, p in enumerate(parsed)]

    def _p50_latency(self):
        with self.metrics.lock:
            recent = self.metrics.latency_ms[-100:]
        if not recent:
            return 0.0
        recent = sorted(recent)
        return recent[len(recent) // 2]

    def _forward_cached(self, ids):
        """Forward only the new suffix; reuse KV + last logits for repeats."""
        if not ids:
            return torch.zeros(self.engine.model.lm_head.weight.shape[0],
                               device=self.engine.device), False
        key = tuple(ids)
        cached = self.logits_cache.get(key)
        if cached is not None:
            self.logits_cache.move_to_end(key)
            return cached, True
        common = 0
        if self.cache_kv is not None and self.cache_tokens:
            limit = min(len(ids), len(self.cache_tokens))
            while common < limit and ids[common] == self.cache_tokens[common]:
                common += 1
        if common == len(ids) and self.cache_logits is not None:
            return self.cache_logits, True

        if common == 0:
            self.cache_kv = None
        elif common < len(self.cache_tokens) and hasattr(self.cache_kv, "crop"):
            self.cache_kv.crop(common)

        new_ids = ids[common:]
        tokens = torch.tensor([new_ids], device=self.engine.device)
        with torch.no_grad():
            logits, kv = self.engine.backbone_logits(tokens, self.cache_kv)
        self.cache_tokens = list(ids)
        self.cache_kv = kv
        self.cache_logits = logits[0]
        self.logits_cache[key] = self.cache_logits
        while len(self.logits_cache) > self.logits_cache_size:
            self.logits_cache.popitem(last=False)
        return self.cache_logits, common > 0

    def take_shown(self, max_age=90.0):
        """P0-3: the ghost text the IME actually displayed, if still fresh."""
        with self.lock:
            shown = self.last_shown
            self.last_shown = None
        if not shown:
            return None
        if time.time() - shown[2] > max_age:
            return None
        return shown

    def save_epoch(self):
        if self.ckpt.save_epoch(lock=self.lock):
            # [P1-9] another process (offline training) had a newer head and we
            # just adopted it, so every cached logit / KV entry is stale now
            self.reset_cache()

    def prefetch_top1(self, ids, top_ids):
        """[P3-4] Warm the LRU with the top-1 continuation.

        The engine walks the most probable chain first, so pre-computing the
        next hop turns the whole chain into cache hits.
        """
        if not self.args.prefetch or not ids or not top_ids:
            return
        child = list(ids) + [int(top_ids[0])]
        if tuple(child) in self.logits_cache:
            return

        def work():
            with self.lock:
                try:
                    self._forward_cached(child)
                except Exception:
                    pass

        threading.Thread(target=work, daemon=True).start()

    def reset_cache(self):
        with self.lock:
            self.cache_tokens = []
            self.cache_kv = None
            self.cache_logits = None
            # weights changed -> every cached logit is stale
            self.logits_cache.clear()

    # ---------- [TRAIN-025] same training mode as the offline pass ----------
    def train_sequence_step(self, context, typed):
        """Per-token rank rewards over a KV-cached walk - exactly the routine the
        offline trainer uses, so both paths score and update the same way."""
        with self.lock:
            t0 = time.perf_counter()
            steps = hits = 0
            loss_sum = 0.0
            try:
                steps, hits, loss_sum = self.engine.train_sequence(
                    context, typed, k=self.args.topk,
                    grad_clip=self.args.grad_clip)
            except Exception as exc:
                print(f"[RL] sequence step failed: {exc!r}", flush=True)
            elapsed = time.perf_counter() - t0
            self.reset_cache()
            return steps, hits, loss_sum, elapsed

    # ---------- S5: guarded RL step ----------
    def rl_step(self, context, typed, reward, negative=False):
        with self.lock:
            t0 = time.perf_counter()
            if negative:
                loss = self.engine.rl_update_unlikelihood(
                    context, typed, reward, self.args.grad_clip)
            else:
                loss = self.engine.rl_update(context, typed, reward,
                                             self.args.grad_clip)
            elapsed = time.perf_counter() - t0
            if elapsed > self.args.step_timeout:
                # [TRAIN-028] This used to "undo" the step by adding back
                # lr * grad - which is only the SGD update rule. With AdamW the
                # applied step is a normalised one, so that would corrupt the
                # weights instead of reverting them. AdamW also bounds each step
                # by its learning rate, so a slow step is no longer dangerous;
                # just record it.
                print(f"[RL] slow step {elapsed*1000:.0f}ms (kept, AdamW)",
                      flush=True)
            self.reset_cache()
            return loss, elapsed

    # ---------- S5: regression probe ----------
    def probe_quality(self):
        with self.lock:
            total = 0.0
            for context, target in self.probe:
                ids = self.engine.tokenizer.encode(context, add_special_tokens=False)
                logits, _ = self._forward_cached(ids)
                probs = F.softmax(logits.float(), dim=-1)
                tid = self.engine.tokenizer.encode(target, add_special_tokens=False)[0]
                total += float(torch.log(probs[tid] + 1e-12))
            self.reset_cache()
            return total / len(self.probe)

    def maybe_rollback(self, ckpt_path):
        quality = self.probe_quality()
        if self.baseline is None:
            self.baseline = quality
            return False
        if quality < self.baseline - self.args.rollback_margin:
            if self.engine.load_checkpoint(ckpt_path):
                self.baseline = self.probe_quality()
                self.metrics.bump("rollbacks")
                self.reset_cache()
                return True
        self.baseline = max(self.baseline, quality)
        return False


class RLLoop(threading.Thread):
    """S2: context log -> reward / backspace negative -> SGD -> checkpoints."""

    def __init__(self, service, args):
        super().__init__(daemon=True)
        self.service = service
        self.args = args
        self.running = True
        self.prev_context = None
        self.prev_tree = None
        self.recent_trees = []
        self.commits = 0
        self.since_eval = 0
        self.pending = []

    def run(self):
        log_file = os.path.abspath(self.args.log_file)
        if not os.path.exists(log_file):
            open(log_file, "a", encoding="utf-8").close()
        pos = os.path.getsize(log_file)
        engine = self.service.engine
        while self.running:
            time.sleep(0.15)
            try:
                size = os.path.getsize(log_file)
            except OSError:
                continue
            if size < pos:
                # the hook restarted / the log was rotated
                pos = 0
            if size <= pos:
                continue
            # [CTX-002] complete lines only, and advance by what we consumed -
            # see the same fix in offline_recorder.run().
            with open(log_file, "rb") as f:
                f.seek(pos)
                data = f.read()
            cut = data.rfind(b"\n")
            if cut < 0:
                continue
            chunk = data[:cut + 1].decode("utf-8", "replace")
            pos += cut + 1
            for line in chunk.splitlines():
                if "[focus]" in line:
                    # P0-1b: a new focused document. Comparing the previous
                    # document's tail against this one produced huge bogus
                    # "replace" negatives, so drop the cross-document state.
                    self.prev_context = None
                    self.prev_tree = None
                    self.recent_trees = []
                    self.service.take_shown()
                    continue
                m = CTX_RE.search(line)
                if not m:
                    continue
                ctx = m.group(1).strip()
                if not ctx:
                    continue
                if len(ctx) > self.args.ctx_chars:
                    ctx = ctx[-self.args.ctx_chars:]
                try:
                    self._on_commit(ctx)
                except Exception as exc:
                    # P1-4: never let one bad commit kill the training thread.
                    print(f"[RL] commit failed: {exc!r}", flush=True)
            self._drain()

    def _on_commit(self, ctx):
        service = self.service
        self.commits += 1
        service.metrics.bump("commits")
        if self.prev_context is not None:
            shown = service.take_shown()          # P0-3
            # [CTX-001] Compare on the overlap, not on the prefix (see
            # ctxwin.py). The hook sends the last N characters, so once the
            # document is longer than N the window slides and prev stops being
            # a prefix of ctx on every plain append - which is how the online
            # loop ended up producing no reward at all for long documents.
            change, changed = ctxwin.classify_change(
                self.prev_context, ctx, self.args.max_change)
            if change == "append" and len(changed) <= self.args.max_accept_chars:
                typed = changed
                reward, path = self._score_accept(typed, shown)
                if reward > 0:
                    service.metrics.bump("rl_rewards")
                # [TRAIN-025] Train on the full typed text, not just the tree
                # path that happened to match, and do it for every commit - the
                # offline pass behaves the same way and a miss is a signal too.
                service.corpus.write({"kind": "accept",
                                      "ctx": self.prev_context,
                                      "typed": typed, "reward": reward,
                                      "shown": bool(shown),
                                      "time": time.time()})
                self._enqueue(("accept", self.prev_context, typed, reward))
            else:
                if change == "append":
                    # a paste or a select-all retype: real text, but not
                    # typing, and one training step per character would hold
                    # the model lock for minutes
                    service.metrics.bump("accept_too_long")
                # P3-1: classify the rejection instead of treating every
                # backspace the same. [CTX-001] hand over the text ctxwin
                # computed, otherwise a long document cannot report a deletion.
                kind, rejected, weight = classify_backspace(
                    self.prev_context, ctx, self.recent_trees,
                    deleted=changed if change == "delete" else None)
                if kind:
                    service.metrics.bump("rl_backspaces")
                    if kind == "predicted-reject":
                        service.metrics.bump("backspace_predicted")
                    elif kind == "typing-reject":
                        service.metrics.bump("backspace_typing")
                    else:
                        service.metrics.bump("backspace_replace")
                    service.corpus.write({"kind": kind, "ctx": ctx,
                                          "rejected": rejected,
                                          "weight": weight,
                                          "time": time.time()})
                    if rejected and len(rejected) <= self.args.max_reject_chars:
                        self._enqueue(("reject", ctx, rejected, weight))
                    elif rejected:
                        service.metrics.bump("reject_too_long")
        # Always resync the tree with the newest context so the next commit can
        # be scored (this is also the positive half of the backspace pairing).
        root, leaves, stats = service.engine.build_tree(
            ctx, self.args.width, self.args.depth)
        if self.commits % 10 == 1:
            print(f"[tree] {stats['time']:.2f}s nodes={stats['nodes']} "
                  f"leaves={stats['leaves']} fwd={stats['forward_calls']} "
                  f"w={self.args.width} d={self.args.depth}", flush=True)
        self.prev_tree = root
        self.recent_trees.append(root)
        if len(self.recent_trees) > 4:
            self.recent_trees = self.recent_trees[-4:]
        self.prev_context = ctx
        self._drain()
        service.save_epoch()

    def _score_accept(self, typed, shown):
        """P0-3: score the ghost text the IME actually showed when available."""
        if shown:
            text, prob, _stamp = shown
            if text:
                n = common_prefix_len(typed, text)
                if n > 0:
                    return prob * (n / len(text)), text[:n]
        reward, path = uw.find_best_reward(self.prev_tree, typed)
        return reward, path

    def _enqueue(self, sample):
        self.pending.append(sample)
        if len(self.pending) > 512:
            self.pending = self.pending[-512:]

    def _drain(self):
        """P0-2: run queued training steps while the model is idle."""
        service = self.service
        while self.pending:
            if (time.time() - service.last_request) < self.args.idle_gate:
                return
            if not self.pending:
                # queue empty -> hand reserved blocks back to the driver
                torch.cuda.empty_cache()
            kind, ctx, text, weight = self.pending.pop(0)
            if kind == "accept":
                steps, hits, lsum, elapsed = service.train_sequence_step(ctx, text)
                if not steps:
                    continue
                service.ckpt.mark_dirty()
                service.metrics.bump("rl_updates", steps)
                print(f"[RL] accept text={text!r} steps={steps} hits={hits} "
                      f"loss={lsum:.3f} {elapsed*1000:.0f}ms "
                      f"pending={len(self.pending)}", flush=True)
                continue
            try:
                loss, elapsed = service.rl_step(ctx, text, weight, negative=True)
            except Exception as exc:
                print(f"[RL] step failed: {exc!r}", flush=True)
                continue
            if loss is None:
                continue
            service.ckpt.mark_dirty()
            service.metrics.bump("rl_updates")
            print(f"[RL] reject text={text!r} w={weight:.4f} "
                  f"loss={loss:.4f} {elapsed*1000:.0f}ms "
                  f"pending={len(self.pending)}", flush=True)
        self.since_eval += 1
        if self.since_eval >= self.args.eval_every:
            self.since_eval = 0
            stable = os.path.join(service.ckpt.ckpt_dir, "lm_head_t2h.pt")
            if service.maybe_rollback(stable):
                print("[S5] rollback to stable checkpoint", flush=True)

    def stop(self):
        self.running = False


class Handler(BaseHTTPRequestHandler):
    service = None
    metrics = None

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"status": "ok"})
        elif self.path.startswith("/metrics"):
            self._send(200, self.metrics.snapshot())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send(400, {"error": str(exc)})
            return
        if self.path.startswith("/metrics"):
            self._send(200, self.metrics.snapshot())
        elif self.path.startswith("/completion"):
            try:
                self._send(200, self.service.submit(req))
            except Exception as exc:  # never take the IME down
                self._send(500, {"error": repr(exc)})
        elif self.path.startswith("/feedback"):
            kind = str(req.get("kind", ""))
            app = str(req.get("app", "") or "")
            if kind == "accept":
                chars = int(req.get("chars", 0) or 0)
                keys = int(req.get("keys", 0) or 0)
                self.metrics.bump("accepts")
                self.metrics.bump("accepted_chars", chars)
                if keys:
                    self.metrics.keystrokes = max(self.metrics.keystrokes, keys)
                self.metrics.bump_app(app, "accepts")
                self.metrics.bump_app(app, "chars", chars)
            elif kind == "shown":
                self.metrics.bump("shown")
                self.metrics.bump_app(app, "shown")
                text = str(req.get("text", "") or "")
                if text and self.service is not None:
                    with self.service.lock:
                        self.service.last_shown = (
                            text, float(req.get("p", 0.0) or 0.0), time.time())
            elif kind == "reject":
                self.metrics.bump("rejects")
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})


def ghost_mode():
    """Read the user-facing mode switch: %APPDATA%\\Rime\\ghost_mode.txt."""
    path = os.path.join(os.environ.get("APPDATA", ""), "Rime", "ghost_mode.txt")
    try:
        with io.open(path, encoding="utf-8-sig") as f:
            value = f.read().strip().lower()
        if value:
            return value
    except Exception:
        pass
    return "online"


def main():
    ap = argparse.ArgumentParser(description="online unified engine service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--model", default=up.MODEL_PATH)
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--device", default=up.DEVICE)
    ap.add_argument("--log-file", default=os.path.join(HERE, "diag", "exp-run-v02.log"))
    ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "diag", "checkpoints_online"))
    ap.add_argument("--corpus", default=os.path.join(HERE, "diag", "corpus.jsonl"))
    ap.add_argument("--rl-lr", type=float, default=up.LR)
    ap.add_argument("-n", "--width", type=int, default=20)
    ap.add_argument("-d", "--depth", type=int, default=2)
    ap.add_argument("--ctx-chars", type=int, default=100)
    ap.add_argument("--idle-gate", type=float, default=0.3)
    # [CTX-008] 0 = off (see the note in train_text_corpus.py)
    ap.add_argument("--grad-clip", type=float, default=0.0)
    # [TRAIN-025] The live path and the offline pass must score identically -
    # the offline pass only gets to be cheaper about it.
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--step-timeout", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--rollback-margin", type=float, default=0.05)
    ap.add_argument("--blacklist", default="",
                    help="comma separated exe names to never predict for")
    ap.add_argument("--dynamic", dest="dynamic", action="store_true",
                    default=True, help="adapt candidate budget to latency")
    ap.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    ap.add_argument("--latency-budget-ms", type=float, default=120.0)
    ap.add_argument("--min-probs", type=int, default=8)
    ap.add_argument("--keys-per-char", type=float, default=3.5)
    ap.add_argument("--max-reject-chars", type=int, default=24,
                    help="ignore reject samples longer than this (document switches)")
    ap.add_argument("--max-change", type=int, default=ctxwin.DEFAULT_MAX_CHANGE,
                    help="bigger edits are pastes/document switches, not typing")
    ap.add_argument("--max-accept-chars", type=int, default=48,
                    help="never train on an accepted run longer than this")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="max sibling requests merged into one forward")
    ap.add_argument("--batch-wait-ms", type=float, default=8.0)
    ap.add_argument("--batch-timeout", type=float, default=30.0)
    ap.add_argument("--ckpt-fp16", action="store_true",
                    help="store checkpoints as fp16 (default fp32)")
    ap.add_argument("--hit-rate-high", type=float, default=0.8)
    ap.add_argument("--hit-rate-low", type=float, default=0.4)
    ap.add_argument("--prefetch", dest="prefetch", action="store_true",
                    default=False,
                    help="warm the LRU with the top-1 child (off: it holds the "
                         "model lock, so it does not reduce wall-clock latency)")
    args = ap.parse_args()

    if ghost_mode() == "offline":
        print("[S1] offline mode: the online engine stays down "
              "(offline_recorder.py handles this mode)", flush=True)
        return 0

    metrics = Metrics()
    engine = up.TreeEngine(args.model, lr=args.rl_lr, device=args.device,
                           dtype=args.dtype, fp8=args.fp8)
    ckpt = uw.CheckpointManager(
        engine, args.ckpt_dir,
        ckpt_dtype="float16" if args.ckpt_fp16 else "float32")
    if ckpt.load_latest():
        print(f"[S1] resumed updates={ckpt.updates}", flush=True)
    corpus = CorpusWriter(args.corpus)
    metrics.keys_per_char = args.keys_per_char
    service = EngineService(engine, ckpt, corpus, metrics, args)

    Handler.service = service
    Handler.metrics = metrics
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"[S1] port {args.port} already in use ({exc}); "
              f"another engine is running, exiting", flush=True)
        return 0
    service.start_batcher()
    loop = RLLoop(service, args)
    loop.start()
    print(f"[S1] online engine listening on http://{args.host}:{args.port} "
          f"(width={args.width} depth={args.depth} dtype={args.dtype})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        # [TRAIN-036] shutting down only guarantees the realtime slot; forcing
        # all five would stamp the 2h/12h rollback points with the current head.
        ckpt.save_epoch(force=True, slots=("lm_head_t0.pt",))
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
