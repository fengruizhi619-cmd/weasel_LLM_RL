#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cli_emojiless_exp_v0.2 - unified PyTorch pipeline.

One model does everything: candidate generation + RL updates.
No llama-server, no HTTP, no save_state/load_state.
Backbone frozen, lm_head trainable, RL from real typing signals.
"""

import argparse, math, os, re, sys, time, unicodedata, threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
MODEL_PATH = (os.environ.get("WEASEL_LLM_MODEL", "").strip()
              or os.path.join(_REPO_ROOT, "models", "Qwen3-0.6B-Base"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
WIDTH = 5
DEPTH = 5
TOP_N = 10
CTX_CHARS = 100


class Node:
    __slots__ = ("tok","p","cum","parent","children","is_leaf","stop","depth")
    def __init__(self, tok="", p=1.0, cum=1.0, parent=None, depth=0):
        self.tok=tok; self.p=p; self.cum=cum
        self.parent=parent; self.children=[]
        self.is_leaf=False; self.stop=""; self.depth=depth
    @property
    def path(self):
        parts=[]; n=self
        while n and n.parent: parts.append(n.tok); n=n.parent
        return "".join(reversed(parts))
    @property
    def is_root(self): return self.parent is None


def rank_reward(rank, total, scheme="harmonic"):
    """[TRAIN-024] Reward for a token the model ranked |rank| (0-based) among the
    |total| reachable candidates at that step.

    Rank the reachable candidates by cumulative probability, pay the token by
    where it landed, and add the per-token rewards up along the sequence - so a
    longer correct run is worth MORE, not less. The old score multiplied
    probabilities together (cum ~ 0.3**10 = 6e-6 for ten correct characters),
    which made long hits invisible next to a single-character hit.

      harmonic : 1, 1/2, 1/3, 1/4, 1/5 ...
      linear   : (total-rank)/total
      exp      : 1, 1/2, 1/4, 1/8 ...
    """
    r = max(0, int(rank))
    if scheme == "linear":
        return float(max(0, int(total) - r)) / float(max(1, int(total)))
    if scheme == "exp":
        return 2.0 ** (-r)
    return 1.0 / float(r + 1)


def is_punct(t):
    return any(unicodedata.category(c).startswith("P") for c in t)

def is_eos_like(t):
    lo=t.lower()
    return "endoftext" in lo or "eos" in lo or "<|im_end|>" in lo or t in("\n","\r\n")


class FP8DynamicLinear(torch.nn.Module):
    """Per-tensor dynamic FP8 Linear for this Windows/Ada PyTorch build."""

    def __init__(self, source: nn.Linear):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        if self.in_features % 16 or self.out_features % 16:
            raise ValueError(
                f"FP8 _scaled_mm requires dimensions divisible by 16: "
                f"{self.in_features}x{self.out_features}")

        weight = source.weight.data.to(torch.float32)
        weight_scale = (weight.abs().amax() / 448.0).clamp_min(1e-12)
        weight_fp8 = (weight / weight_scale).clamp(-448.0, 448.0).to(
            torch.float8_e4m3fn)

        self.register_buffer("weight_fp8", weight_fp8)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        if source.bias is not None:
            self.register_buffer("bias", source.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.in_features).to(torch.float32)
        scale = (flat.abs().amax() / 448.0).clamp_min(1e-12)
        x_fp8 = (flat / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        out = torch._scaled_mm(
            x_fp8, self.weight_fp8.t(), scale_a=scale,
            scale_b=self.weight_scale, bias=self.bias,
            out_dtype=x.dtype, use_fast_accum=True)
        return out.reshape(*shape[:-1], self.out_features)


def replace_backbone_linear_fp8(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FP8DynamicLinear(child))
            count += 1
        else:
            count += replace_backbone_linear_fp8(child)
    return count


class TreeEngine:
    """Unified PyTorch pipeline: generate candidates + RL update, single model."""

    def __init__(self, model_path, lr=1e-4, device="cuda", dtype="float32",
                 fp8=False):
        print(f"[engine] loading model: {model_path} dtype={dtype}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, dtype)).to(device)
        self.device = device

        if fp8:
            count = replace_backbone_linear_fp8(self.model.model)
            print(f"[engine] frozen backbone quantized to dynamic FP8: "
                  f"{count} Linear layers", flush=True)

        # Keep a trainable FP32 head even when the frozen backbone is BF16.
        if self.model.lm_head.weight is self.model.model.embed_tokens.weight:
            self.model.lm_head.weight = nn.Parameter(
                self.model.model.embed_tokens.weight.data.to(torch.float32))
            print("[engine] lm_head untied from embed_tokens as fp32", flush=True)
        else:
            self.model.lm_head.weight = nn.Parameter(
                self.model.lm_head.weight.data.to(torch.float32))
            print("[engine] lm_head promoted to fp32", flush=True)

        # freeze all except lm_head
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.lm_head.weight.requires_grad = True
        self.optimizer = torch.optim.SGD(
            [self.model.lm_head.weight], lr=lr)
        self.model.eval()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[engine] ready on {device}: {total:,} total, "
              f"{trainable:,} trainable (lm_head)", flush=True)
        if device.startswith("cuda"):
            print(f"[engine] cuda memory: "
                  f"{torch.cuda.memory_allocated()/1024**2:.0f} MiB allocated, "
                  f"{torch.cuda.memory_reserved()/1024**2:.0f} MiB reserved",
                  flush=True)

    def backbone_logits(self, input_ids, past_key_values=None):
        """Run the frozen backbone and apply the trainable FP32 head."""
        out = self.model.model(
            input_ids=input_ids, past_key_values=past_key_values,
            use_cache=True)
        hidden = out.last_hidden_state[:, -1, :]
        logits = self.model.lm_head(
            hidden.to(self.model.lm_head.weight.dtype))
        return logits, out.past_key_values

    def get_candidates(self, context_text, k=5):
        """Query model for top-k next token candidates."""
        input_ids = self.tokenizer.encode(context_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits, _ = self.backbone_logits(input_ids)
            probs = F.softmax(logits[0], dim=-1)
            top_p, top_id = torch.topk(probs, k)
        results = []
        for i in range(top_id.shape[0]):
            tok = self.tokenizer.decode(top_id[i].item())
            results.append({"tok": tok, "p": top_p[i].item(),
                           "id": top_id[i].item()})
        return results

    def select_cache(self, cache, indices):
        """Select active branches from a batched DynamicCache."""
        if not indices:
            return DynamicCache()
        device = next((layer.keys.device for layer in cache.layers
                       if getattr(layer, "is_initialized", False)), self.device)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        result = DynamicCache()
        for layer in cache.layers:
            if not getattr(layer, "is_initialized", False):
                continue
            new_layer = DynamicLayer()
            new_layer.dtype = layer.dtype
            new_layer.device = layer.device
            new_layer.keys = layer.keys.index_select(0, index_tensor).contiguous()
            new_layer.values = layer.values.index_select(0, index_tensor).contiguous()
            new_layer.is_initialized = True
            result.layers.append(new_layer)
        return result

    def build_tree(self, context_text, width, depth):
        """Build a candidate tree with one batched forward per tree level."""
        t0 = time.monotonic()
        input_ids = self.tokenizer.encode(context_text, return_tensors="pt").to(self.device)

        root = Node(depth=0)
        stats = {"nodes": 1, "leaves": 0, "forward_calls": 0,
                 "forward_s": 0.0, "select_s": 0.0}

        def candidates_from_logits(values):
            if values.dim() == 1:
                values = values.unsqueeze(0)
            probs = F.softmax(values, dim=-1)
            top_p, top_id = torch.topk(probs, width, dim=-1)
            return [
                list(zip(ids.tolist(), scores.tolist()))
                for ids, scores in zip(top_id, top_p)
            ]

        def add_child(parent, token_id, p):
            tok = self.tokenizer.decode([token_id])
            if not tok:
                return None, False
            child = Node(tok=tok, p=float(p), cum=parent.cum * float(p),
                         parent=parent, depth=parent.depth + 1)
            parent.children.append(child)
            stats["nodes"] += 1

            if is_eos_like(tok):
                child.is_leaf = True; child.stop = "eos"; stats["leaves"] += 1
                return child, False
            if is_punct(tok):
                child.is_leaf = True; child.stop = "punct"; stats["leaves"] += 1
                return child, False
            if child.depth >= depth:
                child.is_leaf = True; child.stop = "depth"; stats["leaves"] += 1
                return child, False
            return child, True

        tf = time.perf_counter()
        with torch.no_grad():
            warm_logits, cache = self.backbone_logits(input_ids)
            root_candidates = candidates_from_logits(warm_logits[0])[0]
        base_len = cache.get_seq_length()

        # Each item is (child_node, child_token_id, parent_cache_row).
        growing = []
        for token_id, p in root_candidates:
            tid = int(token_id)
            child, keep = add_child(root, tid, p)
            if child is not None and keep:
                growing.append((child, tid, 0))

        for _ in range(1, depth):
            if not growing:
                break
            tc = time.perf_counter()
            cache = self.select_cache(cache, [item[2] for item in growing])
            stats["select_s"] += time.perf_counter() - tc

            tokens = torch.tensor([[item[1]] for item in growing],
                                  device=self.device)
            tf = time.perf_counter()
            with torch.no_grad():
                logits, cache = self.backbone_logits(tokens, cache)
            candidate_rows = candidates_from_logits(logits)
            stats["forward_s"] += time.perf_counter() - tf
            stats["forward_calls"] += 1

            next_growing = []
            for row, (parent, _, _) in enumerate(growing):
                for token_id, p in candidate_rows[row]:
                    tid = int(token_id)
                    child, keep = add_child(parent, tid, p)
                    if child is not None and keep:
                        next_growing.append((child, tid, row))
            # [TREE-013] beam cap: keep the |width| highest-cum branches per
            # level, otherwise width^depth explodes (20^10 nodes).
            if len(next_growing) > width:
                next_growing.sort(key=lambda item: item[0].cum, reverse=True)
                next_growing = next_growing[:width]
            growing = next_growing

        elapsed = time.monotonic() - t0
        leaves = []
        def collect(node):
            if node.is_leaf and not node.is_root: leaves.append(node)
            for child in node.children: collect(child)
        collect(root)
        leaves.sort(key=lambda n: n.cum, reverse=True)

        stats["time"] = elapsed
        return root, leaves, stats

    def rl_update(self, context_text, typed_text, reward, grad_clip=0.0):
        """RL update: reinforce the model for producing typed_text after context_text.
        Zero-cost: caches hidden state, only lm_head gets gradient."""
        input_ids = self.tokenizer.encode(context_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            hidden = self.model.model(
                input_ids=input_ids).last_hidden_state[0, -1, :]
            if hidden.dtype != torch.float32:
                hidden = hidden.to(self.model.lm_head.weight.dtype)

        self.model.train()
        logits = self.model.lm_head(hidden)
        log_probs = F.log_softmax(logits, dim=-1)

        # target: first token of typed_text
        target_ids = self.tokenizer.encode(typed_text, add_special_tokens=False)
        if not target_ids:
            return 0.0

        loss = -reward * log_probs[target_ids[0]]
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [self.model.lm_head.weight], grad_clip)
        self.optimizer.step()
        self.model.eval()
        return loss.item()

    def rl_update_unlikelihood(self, context_text, rejected_text, weight=1.0,
                               grad_clip=0.0):
        """S2: negative sample -- push DOWN the probability of the rejected text.

        Same zero-cost shape as rl_update: frozen backbone, cached hidden,
        one backward through the fp32 head only.
        """
        input_ids = self.tokenizer.encode(context_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model.model(
                input_ids=input_ids).last_hidden_state[0, -1, :]
            if hidden.dtype != torch.float32:
                hidden = hidden.to(self.model.lm_head.weight.dtype)
        target_ids = self.tokenizer.encode(rejected_text, add_special_tokens=False)
        if not target_ids:
            return 0.0
        self.model.train()
        logits = self.model.lm_head(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        # minimising log p(rejected) == gradient ascent on the negative likelihood
        loss = weight * log_probs[target_ids[0]]
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [self.model.lm_head.weight], grad_clip)
        self.optimizer.step()
        self.model.eval()
        return loss.item()

    def train_sequence(self, prompt_text, target_text, k=20, grad_clip=1.0,
                       miss_weight=0.3, dry_run=False, scheme="harmonic"):
        """[TRAIN-020 TOPK] Rank-scored RL over one run of real text.

        Training does not need the display tree. One frozen-backbone prefill,
        then one single-token decode per character: for every step the trainable
        head is asked for its top-k next characters - those are the reachable
        candidates - and the character the user actually typed is scored by its
        RANK among them (see rank_reward). Every token gets its own reward and
        its own step, so the rewards simply add up along the sequence and longer
        correct runs are worth more.

        A miss still produces a step - the wrong top-1 is pushed down - so no
        recorded text is thrown away.

        The backbone is frozen, so its KV cache stays valid across the head
        updates and the walk costs one token per step.

        Returns (steps, hits, reward_sum).
        """
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        if not prompt_ids or not target_ids:
            return 0, 0, 0.0

        steps = hits = 0
        reward_sum = 0.0
        head_dtype = self.model.lm_head.weight.dtype

        with torch.no_grad():
            out = self.model.model(
                input_ids=torch.tensor([prompt_ids], device=self.device),
                use_cache=True)
            hidden = out.last_hidden_state[0, -1, :]
            past = out.past_key_values

        self.model.train()
        for target_id in target_ids:
            logits = self.model.lm_head(hidden.to(head_dtype).unsqueeze(0))[0]
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                probs = log_probs.detach().exp()
                top_p, top_id = torch.topk(probs, min(k, probs.shape[-1]))
                match = (top_id == target_id).nonzero(as_tuple=True)[0]
                rank = int(match[0].item()) if match.numel() else -1
                if rank >= 0:
                    reward = rank_reward(rank, int(top_id.shape[0]), scheme)
                else:
                    reward = 0.0

            if rank >= 0:
                hits += 1
                reward_sum += reward
                loss = -reward * log_probs[target_id]
            else:
                loss = miss_weight * log_probs[int(top_id[0].item())]

            steps += 1
            if not dry_run:
                self.optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [self.model.lm_head.weight], grad_clip)
                self.optimizer.step()

            with torch.no_grad():
                out = self.model.model(
                    input_ids=torch.tensor([[int(target_id)]], device=self.device),
                    past_key_values=past, use_cache=True)
                hidden = out.last_hidden_state[0, -1, :]
                past = out.past_key_values
        self.model.eval()
        return steps, hits, reward_sum

    def save_checkpoint(self, path, updates=0, dtype="float32", lock=None):
        """P1-5: keep the master head in fp32; P1-8: copy under the engine lock."""
        target = torch.float16 if dtype == "float16" else torch.float32
        if lock is not None:
            with lock:
                w = self.model.lm_head.weight.detach().to(target).cpu().clone()
        else:
            w = self.model.lm_head.weight.detach().to(target).cpu().clone()
        torch.save({"lm_head_weight": w, "updates": updates, "dtype": dtype,
                    "timestamp": time.strftime("%Y%m%d_%H%M%S")}, path)
        print(f"[engine] checkpoint saved to {path} updates={updates} "
              f"dtype={dtype}")

    def load_checkpoint(self, path):
        if not os.path.exists(path): return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.lm_head.weight.data.copy_(ckpt["lm_head_weight"].float())
        print(f"[engine] checkpoint loaded from {path}")
        return True


def main():
    ap = argparse.ArgumentParser(description="cli_emojiless_exp_v0.2 unified pipeline")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                    default="bfloat16" if DEVICE == "cuda" else "float32")
    ap.add_argument("--fp8", action="store_true",
                    help="quantize frozen backbone Linear layers with TorchAO FP8")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--ctx-chars", type=int, default=CTX_CHARS)
    ap.add_argument("--text", default="", help="prompt text (or interactive)")
    args = ap.parse_args()

    print(f"[v0.2] cli_emojiless_exp_v0.2 unified PyTorch pipeline", flush=True)
    print(f"[v0.2] model: {args.model}", flush=True)
    print(f"[v0.2] device: {args.device} lr={args.lr}", flush=True)

    engine = TreeEngine(args.model, lr=args.lr, device=args.device,
                        dtype=args.dtype, fp8=args.fp8)

    # interactive loop
    print(f"\n[v0.2] commands:", flush=True)
    print(f"  tree <text>    - build candidate tree", flush=True)
    print(f"  rl <ctx> <typed> - RL update (reinforce typed after ctx)", flush=True)
    print(f"  quit           - exit", flush=True)

    while True:
        try:
            line = input("\n[v0.2]> ").strip()
        except EOFError:
            break
        if not line: continue

        if line == "quit" or line == "exit":
            break
        elif line.startswith("tree "):
            text = line[5:].strip()
            if not text: continue
            root, leaves, stats = engine.build_tree(text, args.width, args.depth)
            print(f"[tree] {stats['time']:.2f}s nodes={stats['nodes']} leaves={stats['leaves']} forward_calls={stats.get('forward_calls', 0)} forward={stats.get('forward_s', 0):.2f}s select={stats.get('select_s', 0):.2f}s")
            if leaves:
                print(f"[tree] top {min(3,len(leaves))}:")
                for i, lf in enumerate(leaves[:3], 1):
                    print(f"  {i}. P={lf.cum:.6f} {lf.path!r}")
        elif line.startswith("rl "):
            parts = line[3:].strip().split(" ", 1)
            if len(parts) != 2:
                print("  usage: rl <context> <typed_text>")
                continue
            ctx, typed = parts
            # get candidate probs for ctx
            cands = engine.get_candidates(ctx, 5)
            # check if typed text matches any candidate
            reward = 0.0
            for c in cands:
                if typed.startswith(c["tok"]):
                    reward = c["p"]
                    break
            # RL update
            loss = engine.rl_update(ctx, typed, reward)
            print(f"  [RL] reward={reward:.4f} loss={loss:.6f}")
            # verify: re-query candidates
            new_cands = engine.get_candidates(ctx, 3)
            print(f"  [verify] after update:")
            for c in new_cands:
                print(f"    {c['tok']!r} p={c['p']:.4f}")
        else:
            print(f"  commands: tree <text> | rl <ctx> <typed> | quit")

    print("[v0.2] done")


if __name__ == "__main__":
    main()
