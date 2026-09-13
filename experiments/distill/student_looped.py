# -*- coding: utf-8 -*-
"""循环 Transformer 学生：把 U 个共享层跑 T 轮，等效深度 = U*T。

设计口径（与仓库既有口径一致）：
  - 每轮加一个「轮次嵌入」，否则 pre-norm block 反复映射会收敛到不动点、轮次无法区分
    （见 人工智能算法精英大赛/scripts/train_looped.py 的注释与实测逐轮余弦 0.72->0.905）。
  - 轮次嵌入初始化：默认全零。这样第 1 轮起循环体等价于"同一个 block 反复做几乎相同的事"，
    从"近似静态深栈"出发，由训练自己决定哪几轮特化；随机初始化会让早期轮次被噪声带偏。
  - 层用 transformers 的 Qwen3DecoderLayer（自带 GQA + q_norm/k_norm + SwiGLU），
    与教师逐字段同构，蒸馏时不存在实现差异带来的失真。
  - 早退：每轮循环结束都能出 logits，train_distill.py 会记录"第 t 轮的命中率"。

对照矩阵（train_distill.py --arm 选）：
  u4t7   U=4  T=7   等效深度 28 = 教师深度，参数量也几乎等于教师（基线臂）
  u1t28  U=1  T=28  等效深度 28，参数 1/4（压缩臂）
  u1t8   U=1  T=8   等效深度 8（甜点臂，参数再小）
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm


class LoopedStudent(nn.Module):
    """U 个共享 Qwen3DecoderLayer × T 轮循环 + 轮次嵌入 + lm_head。"""

    def __init__(self, cfg: Qwen3Config, uniq: int, loops: int,
                 round_emb_init: str = "zeros"):
        super().__init__()
        assert uniq >= 1 and loops >= 1
        self.cfg = cfg
        self.uniq = uniq
        self.loops = loops
        self.effective_depth = uniq * loops
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.hidden_size

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg, layer_idx=i) for i in range(uniq)])
        self.round_emb = nn.Embedding(loops, cfg.hidden_size)
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if round_emb_init == "zeros":
            nn.init.zeros_(self.round_emb.weight)
        else:
            nn.init.normal_(self.round_emb.weight, std=0.02)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # 全局初始化必须放在**最后**：self.apply() 会重新初始化 round_emb，
        # 把上面的"零初始化"覆盖成 normal(0,0.02)（实测 round_emb 变成 0.064），
        # 于是每轮循环的输入都被扰动 —— 表现是"手动逐层循环与教师完全一致，
        # 而 forward 只有 0.69 一致率"。这里在 apply 之后再把 round_emb 按原方案重置。
        self.apply(self._init_weights)
        if round_emb_init == "zeros":
            nn.init.zeros_(self.round_emb.weight)
        else:
            nn.init.normal_(self.round_emb.weight, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rotary(self, hidden_states, position_ids):
        """现算 cos/sin。形状必须与 q/k 一致：(B, L, head_dim)。

        踩过的坑：早期写成 (1, L, head_dim) 再交给 apply_rotary_pos_emb，
        在该实现里不广播，直接报
        `The size of tensor a (16) must match the size of tensor b (8)`。
        这里显式按 batch 展开成 (B, L, head_dim)。
        """
        cfg = self.cfg
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        inv_freq = 1.0 / (cfg.rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32,
                         device=hidden_states.device) / head_dim))
        pos = position_ids.float()                          # (B, L)
        freqs = torch.einsum("bl,d->bld", pos, inv_freq)    # (B, L, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)             # (B, L, head_dim)
        return emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype)

    def forward(self, input_ids, attention_mask=None, collect_rounds=False,
                collect_hidden=False, grad_rounds=None):
        """返回 (final_logits, aux)。

        aux = None，或 {"round_logits": [...], "round_hidden": [...]}

        collect_rounds : 收每轮的早退 logits（**无梯度**，评测用；算全轮很贵）
        collect_hidden : 收每轮循环边界的 hidden（**带梯度**，插入式蒸馏用；很便宜）
        grad_rounds    : 只对这些轮号（1 基）额外算**带梯度**的 logits，返回在
                         aux["round_logits_grad"] 里。为什么要限定：每轮 logits 是
                         (B,L,V)=B·L·151936，u1t28 全轮带梯度会吃掉好几 GB；
                         而插入式蒸馏只需要末尾若干轮的输出监督。
        """
        B, L = input_ids.shape
        dev = input_ids.device
        position_ids = torch.arange(L, dtype=torch.long, device=dev).unsqueeze(0).expand(B, L)

        # —— 两个踩过的坑，都在这一小段 ——
        # 1) cos/sin 传 (B, L, head_dim) 的**三维**形状，不要自己补 heads 维：
        #    transformers 的 apply_rotary_pos_emb 内部会做 unsqueeze(unsqueeze_dim=1)，
        #    自己先补一次会变成 5 维，报 "too many values to unpack (expected 4)"。
        # 2) 也**不要**传 attention_mask 给层：torch 2.5.1 的 use_gqa_in_sdpa 要求
        #    mask is None 才启用 GQA 内核；传了会回退 repeat_kv 路径。
        #    训练里序列等长、无 padding，本来就不需要 mask（因果掩码层内处理）。
        #    真要批不等长样本时，改成左对齐 + 逐样本定长切分，别退回传 mask。
        cos, sin = self._rotary(self.embed_tokens.weight[:1], position_ids)

        h = self.embed_tokens(input_ids)
        mask = None
        if attention_mask is not None and not bool(attention_mask.all()):
            causal = torch.full((L, L), torch.finfo(h.dtype).min,
                                dtype=h.dtype, device=dev).triu(1)
            pad = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
            mask = causal[None, None] + pad

        round_logits = []
        round_hidden = []
        round_logits_grad = {}
        want_grad = set(grad_rounds or ())
        for t in range(self.loops):
            h = h + self.round_emb(torch.full((B,), t, dtype=torch.long, device=dev))[:, None, :]
            for li in range(self.uniq):
                # Qwen3DecoderLayer.forward 返回**裸 tensor**（源码就是 `return hidden_states`），
                # 不是 tuple。写 [0] 会切掉批次维（(1,L,D) -> (L,D)）；当 B=1 且
                # D 恰好等于原 L 时，后面的 round_emb 广播会把它"修"回三维，
                # 于是单层配置看起来正常、多层配置在第二层炸 —— 别加 [0]。
                h = self.layers[li](h, attention_mask=mask, position_ids=position_ids,
                                    position_embeddings=(cos, sin))
            # 循环边界：这里是插入式蒸馏的监督点。h 是**未归一化**的，
            # 与教师 hidden_states[0..L-1] 同一约定，可以直接对齐。
            if collect_hidden:
                round_hidden.append(h)
            if collect_rounds:
                with torch.no_grad():
                    round_logits.append(self.lm_head(self.norm(h)).float())
            if (t + 1) in want_grad:
                # 带梯度的早退读出：插入式蒸馏在末几轮做输出级监督。
                round_logits_grad[t + 1] = self.lm_head(self.norm(h))
        # final norm 只施加一次！Qwen3Model.forward 的最后一句就是
        # `hidden_states = self.norm(hidden_states)`，输出已是归一化后的，
        # 而 Qwen3ForCausalLM 直接 lm_head(hidden_states) 不再归一化。
        # 这里若写 self.lm_head(self.norm(h)) 就是**重复归一化**：
        # RMSNorm 不幂等，表现是输出分布看着正常（std 接近）但逐位 argmax 只对 ~0.28。
        logits = self.lm_head(self.norm(h))
        aux = None
        if collect_rounds or collect_hidden or round_logits_grad:
            aux = {"round_logits": round_logits, "round_hidden": round_hidden,
                   "round_logits_grad": round_logits_grad}
        return logits, aux

    # ---- 工具 ----
    def param_report(self):
        emb = sum(p.numel() for p in self.embed_tokens.parameters())
        blk = sum(p.numel() for p in self.layers.parameters())
        head = self.lm_head.weight.numel()
        tied = self.lm_head.weight is self.embed_tokens.weight
        others = sum(p.numel() for n, p in self.named_parameters()
                     if not n.startswith(("embed_tokens", "layers")) and "lm_head" not in n)
        total = sum(p.numel() for p in self.parameters())
        # 踩过：`parameters()` 对共享（tie）参数**本来就只产出一次**（_named_members
        # 默认 remove_duplicate=True），所以 total 已经是唯一参数量。先前这里又减了一次
        # emb，导致所有 tie 模型的"唯一合计"少报 155,582,464 —— blank28/stack28 被
        # 报成 440.5M（真值 596.05M，与教师等参）。
        real = total
        return {"embed": emb, "blocks": blk, "lm_head": head, "others": others,
                "tied": tied, "total_unique": real}

    @torch.no_grad()
    def init_from_teacher(self, teacher_state: dict):
        """把教师权重填进学生：U 个共享层轮流吃掉教师的前 U 层（其余丢弃）；
        嵌入原样搬运（形状相同：词表与 d_model 都没变）。

        注意 tie 的语义：`tie_word_embeddings=true` 的模型（教师 Qwen3-0.6B 即如此）
        **state_dict 里仍然有 lm_head.weight**（311 个键、合计 751,632,384），
        它和 embed_tokens.weight 指向同一个张量；而 `parameters()` 只数一次（596,049,920）。
        两个数不一致是 tie 的正常表现，不是 bug。所以搬 embed_tokens 就等于同时搬了头。
        """
        missing, used = [], set()
        sd = self.state_dict()
        # 教师（Qwen3ForCausalLM）的键带 `model.` 前缀，学生不带。
        # 踩过：只查 'embed_tokens.weight' / 'norm.weight' 必然未命中，
        # 学生开局就带着随机嵌入与随机头（head 与 embed tie），
        # 表现是 agree_top1 恒为 0、kl 好几百 —— 其实是初始化没生效，不是架构不行。
        for tkey, skey in (("model.embed_tokens.weight", "embed_tokens.weight"),
                           ("model.norm.weight", "norm.weight")):
            if tkey in teacher_state and skey in sd \
                    and sd[skey].shape == teacher_state[tkey].shape:
                sd[skey].copy_(teacher_state[tkey]); used.add(skey)
            else:
                missing.append(f"{tkey} -> {skey}")
        # 教师若是 untie 的（state_dict 里有 lm_head.weight）则一并搬
        if "lm_head.weight" in teacher_state and "lm_head.weight" in sd \
                and sd["lm_head.weight"].shape == teacher_state["lm_head.weight"].shape:
            sd["lm_head.weight"].copy_(teacher_state["lm_head.weight"]); used.add("lm_head.weight")
        # 共享层：第 i 个学生层吃教师第 i 层
        for i in range(self.uniq):
            for suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                           "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                           "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                           "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                           "input_layernorm.weight", "post_attention_layernorm.weight"):
                tk = f"model.layers.{i}.{suffix}"
                sk = f"layers.{i}.{suffix}"
                if tk in teacher_state and sk in sd and sd[sk].shape == teacher_state[tk].shape:
                    sd[sk].copy_(teacher_state[tk]); used.add(sk)
                else:
                    missing.append(sk)
        return used, missing

    @torch.no_grad()
    def tie_follows(self):
        """tie 时把 lm_head 指回 embedding（init 之后调用，防止 copy_ 把绑定关系弄丢）。"""
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight


def build_student(teacher_model_path: str, uniq: int, loops: int,
                  round_emb_init: str = "zeros", attn_impl: str = "sdpa",
                  hidden_size: int | None = None,
                  tie_word_embeddings: bool | None = None) -> LoopedStudent:
    """构造学生。

    默认**完全沿用教师的 config**（d_model、词表、GQA、QK-norm、RoPE 全部一致），
    只把"28 个独立层"换成 U 个共享层跑 T 轮。这一点很关键：
    一旦改动 hidden_size 或词表，嵌入/头就与教师形状不符、权重无法继承
    （冒烟时踩过：d=256 的学生 init 未命中 embed_tokens 与 norm，
     开局 kl=618、agree=0，等于随机初始化，"等效参数"的前提就不成立了）。

    必须显式设置 `_attn_implementation`：裸 Qwen3Config 该字段是 None，
    Qwen3Attention 会去 ALL_ATTENTION_FUNCTIONS[None] 取实现并抛 KeyError；
    只有走 AutoModel 加载路径时它才会被填上。默认 sdpa（CUDA 上最快）。
    """
    cfg = Qwen3Config.from_pretrained(teacher_model_path)
    cfg._attn_implementation = attn_impl
    if hidden_size is not None:
        cfg.hidden_size = hidden_size
        cfg.head_dim = hidden_size // cfg.num_attention_heads
    if tie_word_embeddings is not None:
        cfg.tie_word_embeddings = tie_word_embeddings
    return LoopedStudent(cfg, uniq, loops, round_emb_init=round_emb_init)
