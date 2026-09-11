#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cli_emojiless_exp_v0.2 - unified PyTorch pipeline.

One model does everything: candidate generation + RL updates.
No llama-server, no HTTP, no save_state/load_state.
Backbone frozen, lm_head trainable, RL from real typing signals.
"""

import os, time, unicodedata
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
# Learning rate for the plain-SGD path.
#
# Why not AdamW, since an earlier revision used it: AdamW's step is ~lr per
# parameter whatever the gradient is, so the reward cannot decide how far a
# candidate moves. The design needs update = lr * gradient with the reward
# folded into the gradient (loss = -score * log p), which is what SGD gives:
# a rank-20 candidate moves 1/20 as far as a rank-1 candidate.
#
# [CTX-008] Calibrated on this head (std 0.030, |h| ~ 134, vocab 151936):
#   gradient norm ~ 68 at score 1.0, ~ 3 at score 0.05  ->  grad RMS 5.5e-3
#   consecutive gradients are ORTHOGONAL (mean cosine 0.000), so the total
#   drift is a random walk sqrt(steps) * lr * grad_rms, not steps * ...
#   a 12779-step pass therefore moves ~0.62 * lr; 5e-4 lands at ~3e-4, i.e. 1%
#   of the weight std. That dose is what a full pass over ~24k characters of
#   text comes to.
# WEASEL_LLM_LR overrides it.
LR = float(os.environ.get("WEASEL_LLM_LR", "").strip() or 5e-4)


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


def is_eos_like(t):
    lo=t.lower()
    return "endoftext" in lo or "eos" in lo or "<|im_end|>" in lo or t in("\n","\r\n")


# The ellipsis is punctuation by category but it is NOT a stop: the rule is
# 。，！？;: 之流 end a prediction and "…" can sit inside a sentence, so a branch
# that reaches it keeps growing.
_ELLIPSIS = "…⋯…"


def is_punct(t):
    if not t:
        return False
    if all(c in _ELLIPSIS for c in t):
        return False
    return any(unicodedata.category(c).startswith("P") for c in t)


def is_no_target(t):
    """[TRAIN-036] Everything the tree refuses to grow through is also something
    the model must never be asked to predict - so it can never be a training
    target either. build_tree() stops on exactly is_eos_like() and is_punct(),
    and the trainer has to use the same set or the two disagree about what the
    model is for: punctuation AND newline/eos used to be scored."""
    return is_punct(t) or is_eos_like(t)


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
        # [CTX-008] Plain gradient descent, because the reward has to decide the
        # step size: update = lr * gradient, and the loss is -score * log p, so
        # the gradient - and therefore the step - is proportional to the score.
        # A rank-20 candidate moves 1/20 as far as a rank-1 candidate, which is
        # the whole point of the rank ladder.
        #
        # AdamW was tried first and removed: it divides by the running RMS of the
        # gradient, so multiplying every reward by a constant changes nothing at
        # all and a 20x reward difference is compressed into nearly the same
        # step. It cannot express "the reward decides the step".
        self.optimizer = torch.optim.SGD([self.model.lm_head.weight], lr=lr)
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
        Zero-cost: caches hidden state, only lm_head gets gradient.

        Returns the loss, or None when no step was taken."""
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
            return None
        # [TRAIN-035] Nothing the tree stops on is ever reinforced. None, not
        # 0.0: callers use `loss is None` to mean "no step was taken", so a 0.0
        # made every punctuation accept count as an update.
        if is_no_target(self.tokenizer.decode([int(target_ids[0])])):
            return None
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

    def train_sequence(self, prompt_text, target_text, k=20, grad_clip=0.0,
                       dry_run=False):
        """[TRAIN-020 TOPK] Rank-scored RL over one run of real text.

        Training does not need the display tree. One frozen-backbone prefill,
        then one single-token decode per character: for every position the head
        is asked for its logits once and the loss is the plain cross entropy of
        the token that really comes next.

        Every scored position produces a gradient, so `steps` counts positions
        and `hits` counts the ones whose true token is inside the top-k. A miss is not punished:
        the only thing this design punishes is taking back a prediction that was
        already accepted, which is the backspace path.

        The backbone is frozen, so its KV cache stays valid across the head
        updates and the walk costs one token per step.

        Returns (steps, hits, loss_sum): positions scored, positions whose
        true token landed in the top-k, and the summed cross entropy.
        """
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        if not prompt_ids or not target_ids:
            return 0, 0, 0.0

        steps = hits = 0
        loss_sum = 0.0
        head_dtype = self.model.lm_head.weight.dtype

        with torch.no_grad():
            out = self.model.model(
                input_ids=torch.tensor([prompt_ids], device=self.device),
                use_cache=True)
            hidden = out.last_hidden_state[0, -1, :]
            past = out.past_key_values

        self.model.train()
        for target_id in target_ids:
            tid = int(target_id)
            # [TRAIN-035] Nothing the tree stops on is a target. The model is
            # never asked to guess a 。 or ， or a newline, so scoring one only
            # teaches the head whatever the recorder happened to catch (the
            # sample batch had a single ， 47 times). It stays in the context -
            # the KV cache is still advanced - it just gets no step.
            if is_no_target(self.tokenizer.decode([tid])):
                with torch.no_grad():
                    out = self.model.model(
                        input_ids=torch.tensor([[tid]], device=self.device),
                        past_key_values=past, use_cache=True)
                    hidden = out.last_hidden_state[0, -1, :]
                    past = out.past_key_values
                continue

            logits = self.model.lm_head(hidden.to(head_dtype).unsqueeze(0))[0]
            log_probs = F.log_softmax(logits, dim=-1)

            # [CTX-010] Plain cross entropy on the token that really comes next.
            # The rank ladder is gone. A controlled run over
            # {损失打在模型自己的候选上, 打在真值 token 上} x {正挂, 倒挂, 无权重}
            # (experiments/ladder_ablation.py, same text, same start, same lr)
            # showed the ladder's direction changes nothing measurable, while
            # moving the loss onto the TRUE token is what lifted the hit rate
            # from 66% to 83% and the longest chain from 12 to 23. So: no
            # weighting, no rank, no "best matching candidate".
            loss = -log_probs[tid]
            with torch.no_grad():
                # top-k is kept for reporting only: `hits` is the share of
                # positions whose true token is inside the candidate list.
                probs = log_probs.detach().exp()
                top_p, top_id = torch.topk(probs, min(k, probs.shape[-1]))
                if bool((top_id == tid).any()):
                    hits += 1
            loss_sum += float(loss.detach())
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
                    input_ids=torch.tensor([[tid]], device=self.device),
                    past_key_values=past, use_cache=True)
                hidden = out.last_hidden_state[0, -1, :]
                past = out.past_key_values
        self.model.eval()
        return steps, hits, loss_sum

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

