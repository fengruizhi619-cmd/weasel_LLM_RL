#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct ctypes interface to llama.dll for tree-structured KV cache."""

import ctypes, ctypes.wintypes, os, sys, time, unicodedata
import numpy as np

# [CT-001 DLL]
DLL_PATH = r"E:\llama.cpp\llama.dll"
llama = ctypes.CDLL(DLL_PATH)

# C types
llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

# [CT-002 STRUCTS]
class llamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("emb", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_ids", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

# function prototypes
llama.llama_model_load_from_file.restype = ctypes.c_void_p
llama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
llama.llama_free_model.restype = None
llama.llama_free_model.argtypes = [ctypes.c_void_p]
llama.llama_init_from_model.restype = ctypes.c_void_p
llama.llama_init_from_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
llama.llama_free.restype = None
llama.llama_free.argtypes = [ctypes.c_void_p]
llama.llama_n_ctx.restype = ctypes.c_uint32
llama.llama_n_ctx.argtypes = [ctypes.c_void_p]
llama.llama_n_vocab.restype = ctypes.c_uint32
llama.llama_n_vocab.argtypes = [ctypes.c_void_p]
llama.llama_tokenize.restype = ctypes.c_int32
llama.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                  ctypes.POINTER(llama_token), ctypes.c_int32,
                                  ctypes.c_bool, ctypes.c_bool]
llama.llama_batch_init.restype = llamaBatch
llama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
llama.llama_batch_free.restype = None
llama.llama_batch_free.argtypes = [llamaBatch]
llama.llama_decode.restype = ctypes.c_int32
llama.llama_decode.argtypes = [ctypes.c_void_p, llamaBatch]
llama.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
llama.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
llama.llama_memory_seq_cp.restype = None
llama.llama_memory_seq_cp.argtypes = [ctypes.c_void_p, llama_seq_id, llama_seq_id,
                                       llama_pos, llama_pos]
llama.llama_memory_seq_rm.restype = None
llama.llama_memory_seq_rm.argtypes = [ctypes.c_void_p, llama_seq_id, llama_pos, llama_pos]
llama.llama_get_memory.restype = ctypes.c_void_p
llama.llama_get_memory.argtypes = [ctypes.c_void_p]


# [CT-003 MODEL-CTX]
def load_model(model_path, ctx_size=1024):
    """Load GGUF model and create context. Returns (model_ptr, ctx_ptr)."""
    model_ptr = llama.llama_model_load_from_file(
        model_path.encode("utf-8"), 0  # 0 = default flags
    )
    if not model_ptr:
        raise RuntimeError("cannot load model")

    # llama_context_params: struct with ~15 fields
    # We need to create this struct properly.
    # For simplicity, let's use the default params approach.
    # Actually, llama_init_from_model takes llama_context_params struct.
    # Let me define it:
    class llamaContextParams(ctypes.Structure):
        _fields_ = [
            ("n_ctx", ctypes.c_uint32),
            ("n_batch", ctypes.c_uint32),
            ("n_ubatch", ctypes.c_uint32),
            ("n_seq_max", ctypes.c_uint32),
            ("n_threads", ctypes.c_uint32),
            ("n_threads_batch", ctypes.c_uint32),
            ("rope_scaling_type", ctypes.c_int32),
            ("rope_freq_base", ctypes.c_float),
            ("rope_freq_scale", ctypes.c_float),
            ("yarn_ext_factor", ctypes.c_float),
            ("yarn_attn_factor", ctypes.c_float),
            ("yarn_beta_fast", ctypes.c_float),
            ("yarn_beta_slow", ctypes.c_float),
            ("yarn_orig_ctx", ctypes.c_uint32),
            ("cb_eval", ctypes.c_void_p),
            ("cb_eval_user_data", ctypes.c_void_p),
            ("type_k", ctypes.c_int32),
            ("type_v", ctypes.c_int32),
            ("logits_all", ctypes.c_bool),
            ("embeddings", ctypes.c_bool),
            ("offload_kqv", ctypes.c_bool),
            ("flash_attn", ctypes.c_bool),
            ("abort_callback", ctypes.c_void_p),
            ("abort_callback_data", ctypes.c_void_p),
        ]

    params = llamaContextParams()
    params.n_ctx = ctx_size
    params.n_batch = 256
    params.n_ubatch = 256
    params.n_seq_max = 8  # support 8 parallel sequences for tree branching
    params.n_threads = 4
    params.n_threads_batch = 4
    params.logits_all = False  # we only need last position
    params.embeddings = False
    params.offload_kqv = True
    params.flash_attn = True

    ctx_ptr = llama.llama_init_from_model(model_ptr, ctypes.byref(params))
    if not ctx_ptr:
        raise RuntimeError("cannot create context")

    return model_ptr, ctx_ptr


def tokenize(llm_model_ptr, ctx_ptr, text):
    """Tokenize text. Returns list of token IDs."""
    text_bytes = text.encode("utf-8")
    n_max = len(text) * 4 + 16
    tokens_buf = (llama_token * n_max)()
    n = llama.llama_tokenize(
        llm_model_ptr, text_bytes, len(text_bytes),
        tokens_buf, n_max, False, True  # add_special=False, parse_special=True
    )
    if n < 0:
        raise RuntimeError(f"tokenize error: {n}")
    return list(tokens_buf[:n])


def create_single_token_batch(token_id, pos, seq_id):
    """Create a batch for decoding a single token into a specific sequence."""
    batch = llama.llama_batch_init(1, 0, 1)
    # Set fields manually
    batch.n_tokens = 1
    batch.token[0] = token_id
    batch.pos[0] = pos
    batch.n_seq_id[0] = 1
    batch.seq_ids[0][0] = seq_id
    batch.logits[0] = 1  # we want logits for this token
    return batch


# [CT-004 TREE-BUILDER]
class LlamaTreeBuilder:
    def __init__(self, model_path, ctx_size=1024):
        self.model_ptr, self.ctx_ptr = load_model(model_path, ctx_size)
        self.n_vocab = llama.llama_n_vocab(self.model_ptr)
        self.n_ctx = llama.llama_n_ctx(self.ctx_ptr)
        self.mem_ptr = llama.llama_get_memory(self.ctx_ptr)
        print(f"[ct] n_vocab={self.n_vocab} n_ctx={self.n_ctx}")

    def tokenize(self, text):
        return tokenize(self.model_ptr, self.ctx_ptr, text)

    def decode_prompt(self, tokens, seq_id=0):
        """Decode a sequence of tokens into the specified sequence."""
        # Use llama_batch_init for the full prompt
        batch = llama.llama_batch_init(len(tokens), 0, 1)
        for i, tid in enumerate(tokens):
            batch.token[i] = tid
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            # Set seq_id: need to write to seq_ids[i][0]
            batch.seq_ids[i][0] = seq_id
            batch.logits[i] = 1 if i == len(tokens) - 1 else 0

        ret = llama.llama_decode(self.ctx_ptr, batch)
        llama.llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f"decode error: {ret}")

    def branch(self, src_seq, dst_seq, new_token_id, pos):
        """Branch: copy src_seq KV to dst_seq, then decode new_token into dst_seq."""
        # Copy KV from src to dst (entire sequence)
        llama.llama_memory_seq_cp(self.mem_ptr, src_seq, dst_seq, 0, -1)
        # Decode the new token into dst_seq
        batch = create_single_token_batch(new_token_id, pos, dst_seq)
        ret = llama.llama_decode(self.ctx_ptr, batch)
        llama.llama_batch_free(batch)
        if ret != 0:
            raise RuntimeError(f"branch decode error: {ret}")

    def get_logits(self, pos=-1):
        """Get logits for the last decoded token."""
        ptr = llama.llama_get_logits_ith(self.ctx_ptr, pos)
        if not ptr:
            raise RuntimeError("no logits available")
        arr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float * self.n_vocab))
        return np.frombuffer(arr.contents, dtype=np.float32)

    def free(self):
        if self.ctx_ptr:
            llama.llama_free(self.ctx_ptr)
        if self.model_ptr:
            llama.llama_model_free(self.model_ptr)


# [CT-005 DEMO]
def main():
    import math
    model_path = r"E:\llama.cpp\models\Qwen3-0.6B-Base-Q8_0.gguf"

    print("[tree] loading model...")
    builder = LlamaTreeBuilder(model_path, ctx_size=1024)

    prompt = "明天天气如何，是"
    tokens = builder.tokenize(prompt)
    print(f"[tree] prompt: {prompt!r} -> {len(tokens)} tokens")

    # Decode prompt into seq 0
    t0 = time.monotonic()
    builder.decode_prompt(tokens, seq_id=0)
    print(f"[tree] prompt decoded in {time.monotonic()-t0:.2f}s")

    # Get logits after prompt
    logits = builder.get_logits(-1)
    # Top-5 by probability
    arr = np.frombuffer(logits, dtype=np.float32)[:builder.n_vocab]
    probs = np.exp(arr - arr.max())
    probs /= probs.sum()
    top_idx = np.argsort(probs)[::-1][:5]

    print("\n[candidates]")
    candidates = []
    for i in top_idx:
        tok_bytes = llama.llama_detokenize if hasattr(llama, 'llama_detokenize') else None
        # Use builder's model for detokenize
        tok_buf = (ctypes.c_char * 64)()
        # Actually, let me use a simpler approach - call llama_detokenize or
        # read from the vocab. For now, use the token_id and we can decode later.
        candidates.append({"id": int(i), "p": float(probs[i]), "tok": f"tok_{i}"})
        print(f"  id={i} p={probs[i]:.4f}")

    # Branch: for each of top-2 candidates, create a branch sequence
    for branch_idx in range(2):
        cand_id = int(top_idx[branch_idx])
        branch_seq = branch_idx + 1  # seq 1 and seq 2
        pos = len(tokens)  # next position after prompt

        print(f"\n[branch {branch_seq}] copying seq 0 → seq {branch_seq}, "
              f"decoding token {cand_id}...")

        t0 = time.monotonic()
        builder.branch(seq_id_src=0, dst_seq=branch_seq,
                      new_token_id=cand_id, pos=pos)
        print(f"[branch {branch_seq}] decoded in {time.monotonic()-t0:.2f}s")

        # Get logits after branch decode
        logits = builder.get_logits(-1)
        arr = np.frombuffer(logits, dtype=np.float32)[:builder.n_vocab]
        probs = np.exp(arr - arr.max())
        probs /= probs.sum()
        next_top = np.argsort(probs)[::-1][:3]
        print(f"  next candidates: "
              f"{[(int(i), f'{probs[i]:.4f}') for i in next_top]}")

    builder.free()
    print("\n[tree] done")


if __name__ == "__main__":
    main()
