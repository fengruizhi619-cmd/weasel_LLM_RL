#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""蒸馏实验：循环 Transformer 学生 <- Qwen3-0.6B 教师。

第一步只回答一个问题：**知识能不能迁移过去**。
所以基线臂刻意让学生的"等效深度和参数量都等于教师"，把变量压到只剩架构形状：
    u4t7  U=4 层 × T=7 轮 = 等效深度 28 = 教师深度，参数量也几乎等于教师
在此之上再给压缩臂（u1t28 / u1t8），用来回答"能不能更小"。

用法（云端 Linux，先冒烟再正式）：
  # 1) 冒烟：抽 200 条、跑 20 步，验证通路与显存
  python train_distill.py --arm u4t7 --smoke

  # 2) 基线臂：等效深度与参数量均对齐教师
  nohup python train_distill.py --arm u4t7 --init pretrained --steps 6000 \
        --out runs/distill_u4t7 > logs/u4t7.log 2>&1 &

  # 3) 压缩臂
  nohup python train_distill.py --arm u1t28 --init pretrained --steps 6000 \
        --out runs/distill_u1t28 > logs/u1t28.log 2>&1 &

数据：默认把小说正文与打字记录拼成一条流，按 9:1 切训练/留出（留出取尾部，天然不重叠）。
     打字记录来自 diag/segments.jsonl 的 segment 字段（一条 = 一次提交断点）。
     文本与打字都按字符直接喂，不做分块——蒸馏是逐位置的，不需要 chunk。

评估（每一步都在留出集上算，写进 metrics.jsonl）：
  kl            学生与教师的 KL 散度（蒸馏的直接目标）
  agree_top1    学生 argmax 与教师 argmax 的一致率  <- 迁移是否成立的判据
  student_acc   学生 argmax 等于真实下一个字的比率（任务侧参考）
  round_t_agree 第 t 轮循环的早退输出与教师 top-1 的一致率（早退曲线）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from student_looped import build_student  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm  # noqa: F401,E402

ARMS = {           # 臂名 -> (uniq 层数, loops 轮数)；等效深度 = uniq*loops
    # —— 第一步：只回答"知识能不能迁移过去" ——
    "blank28": (28, 1),     # 【数据充足性对照】与教师**完全同构**但随机初始化，
                            #   用来单独问一句：我们的数据够不够蒸馏？
                            #   它成了 → 数据够用，前面的问题在架构上；它不成 → 瓶颈在数据。
    "stack28": (28, 1),     # 同构 + 逐层继承教师（通路验证臂，开局 agree≈1.0）
    "u4t7": (4, 7),         # 循环：等效深度 28（=教师），参数 63M（教师的 1/12）
    "stack4": (4, 1),       # 不循环对照：等效深度 4、参数 63M（与 u4t7 同参数不同深度）
    # —— 第二步：压缩到多小还能用 ——
    "u1t28": (1, 28),       # 等效深度 28，参数约 16M
    "u1t8": (1, 8),         # 等效深度 8（甜点臂）
    "u1t4": (1, 4),
    # —— 插入式蒸馏：监督点落在"每 U 层一个循环边界"上 ——
    "u2t14": (2, 14),       # 每 2 层一个监督点，等效深度 28（教师深度）
    "u2t5": (2, 5),         # 每 2 层一个监督点，等效深度 10（便宜迭代臂）
    "u1t9": (1, 9),         # 非均匀折叠专用：9 轮，配 --ins-depths 让轮次按教师真实
                            #   计算密度分配（浅层大步、深层小步），9 层前向 vs 教师 28
}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ---------------------------------------------------------------- 数据

def read_text(path):
    """按 UTF-8 / GBK 依次尝试读文本。"""
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit("cannot decode " + path)


def load_segments(path, limit=0):
    """打字记录：每行一条 JSON，取 segment 字段拼成文本。"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            seg = rec.get("segment") or rec.get("text") or ""
            if seg:
                out.append(seg)
            if limit and len(out) >= limit:
                break
    return out


def build_corpus(args):
    """返回 (train_text, holdout_novel, holdout_typing)。

    配比问题（必须显式处理）：小说是叙事文，打字语料大量是技术对话（"现在用新的49条"）。
    混在一起会让"能不能迁移"的结论变糊，所以：
      --mix-typing 0    只用小说（干净基线）
      --mix-typing 0.3  小说为主 + 打字占约 3 成（贴近真实分布）
    两种都跑一遍，才能分清"迁移成立"与"专有数据带来的增益"。
    """
    novel = read_text(args.novel) if os.path.exists(args.novel) else None
    if novel is None:
        if not args.novel:
            raise SystemExit("需要 --novel")
        raise SystemExit("找不到小说文本：" + args.novel)
    log("小说 %s：%d 字" % (os.path.basename(args.novel), len(novel)))

    typing = []
    if args.typing and os.path.exists(args.typing) and args.mix_typing > 0:
        with open(args.typing, encoding="utf-8", errors="replace") as f:
            typing = [ln.strip() for ln in f if ln.strip()]
        log("打字语料 %s：%d 行 / %d 字"
            % (os.path.basename(args.typing), len(typing), sum(map(len, typing))))

    # 打字域留出：末尾 N 行不参与训练，单独当"打字域评测集"
    hold_typing = ""
    if typing and args.holdout_typing > 0:
        k = max(1, int(len(typing) * args.holdout_typing))
        hold_typing = "\n".join(typing[-k:])
        typing = typing[:-k]
        log("打字域留出 %d 行 / %d 字" % (k, len(hold_typing)))

    # 小说留出必须**切在混料之前**。
    # 踩过：早先是先 shuffle 混料、再拿 `text[cut:]` 当小说留出，于是那个"小说留出集"
    # 其实是混料尾部 —— 里面按同样比例掺进了打字语料。症状很好认：教师自己在
    # "小说留出集"上的 top1 会随 --mix-typing 变（实测 0.3252 → 0.2510），
    # 而教师上限本该是常数。打字留出集因为切在混料之前，一直是干净的。
    cut = int(len(novel) * (1 - args.holdout))
    tr_novel, hold_novel = novel[:cut], novel[cut:]
    log("小说留出 %d 字（干净：不含打字语料）" % len(hold_novel))

    # 按字符配比把打字数据摊进小说流（逐行交替插入，不破坏窗口连续性）
    share = min(1.0, max(0.0, args.mix_typing))
    if typing and share >= 0.95:
        # share=1.0 时 1-share=0，旧写法 want_typ 会除爆成 1e14，把小说也全带上，
        # 于是"纯打字"跑出来其实是"小说+全部打字"。这里显式短路。
        text = "\n".join(typing)
        log("只用打字语料（%d 字）" % sum(map(len, typing)))
    elif typing and share > 0:
        novel_parts, typ_parts = [], []
        want_typ = len(tr_novel) * share / max(1e-9, 1 - share)
        acc, i = 0, 0
        n_lines = max(1, len(tr_novel.split("\n")))
        for ln in tr_novel.split("\n"):
            novel_parts.append(ln)
            while i < len(typing) and acc < want_typ * (len(novel_parts) / n_lines):
                typ_parts.append(typing[i]); acc += len(typing[i]); i += 1
        merged = novel_parts + typ_parts
        random.Random(args.seed).shuffle(merged)   # 打散，避免"前半小说后半打字"
        text = "\n".join(merged)
        log("混料：小说 %d 字 + 打字 %d 字 → 打字占比约 %.0f%%"
            % (len(tr_novel), acc, 100.0 * acc / max(1, len(tr_novel) + acc)))
    else:
        text = tr_novel
        log("只用小说（--mix-typing 0）")

    if len(text) < 10000:
        raise SystemExit("语料太小（%d 字）" % len(text))
    return text, hold_novel, hold_typing


def make_windows(tokenizer, text, seq_len, device, limit=0):
    """一条流 → 逐 token 的 (input_ids, labels)，全部等长 seq_len。

    不做分块：窗口之间首尾相接，与在线推理"光标前 N 字 → 预测下一个字"一致。
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if limit:
        ids = ids[:limit]
    n = (len(ids) - 1) // seq_len
    xs, ys = [], []
    for i in range(n):
        a = i * seq_len
        chunk = ids[a:a + seq_len + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


# ---------------------------------------------------------------- 插入式蒸馏

def parse_depths(spec, loops):
    """解析显式的边界深度表：--ins-depths "4,8,12,16,20,22,24,26,28"。

    为什么要这个：教师 28 层的计算**极不均匀**（实测前 20 层纯建特征、后 8 层才预测），
    所以"每轮均匀推进 U 层"本身就是错的分配 —— 9 轮均匀铺 28 层，每轮要做 3 层的活，
    而教师那 3 层的性质在浅层和深层完全不同。显式深度表让轮次按教师的**真实计算密度**
    分配：浅层大步、深层小步。
    """
    if not spec:
        return None
    try:
        ds = [int(x) for x in str(spec).replace("，", ",").split(",") if x.strip()]
    except ValueError:
        raise SystemExit("--ins-depths 必须是逗号分隔的整数，如 4,8,12,16,20,22,24,26,28")
    if len(ds) != loops:
        raise SystemExit("--ins-depths 有 %d 个深度，但该臂是 %d 轮循环，必须一一对应"
                         % (len(ds), loops))
    if any(d < 1 for d in ds) or ds != sorted(ds):
        raise SystemExit("--ins-depths 必须是从小到大、都 >=1 的深度")
    return ds


def boundary_depths(arm_uniq, arm_loops, teacher_layers, mapping="head", explicit=None):
    """学生第 t 轮（1 基）的循环边界该对齐教师第几层。返回 list[loops]。

    mapping="head"：d_t = t · U（从头对齐）—— "学生每 U 层对应教师前 U 层"。
    mapping="tail"：d_t = L - (T - t) · U（从尾对齐）—— 把"轮数不足铺满深度"的臂
        （u1t8: T=8 < 28）锚在教师的**精修段**上，而不是锚在教师的特征构建段上。

    **为什么要留 tail 这个选项（这是测出来的，不是猜的）**：教师逐层 logit-lens
    top1 曲线极不均匀（见 probe_depth.py 实测）——
        小说留出：深度 1~17 → 0.001~0.036；20 → 0.123；24 → 0.248；27 → 0.363；28 → 0.401
        打字留出：深度 1~9  → 0.000~0.013；20 → 0.178；22 → 0.470；26 → 0.795；28 → 0.82
    即**前 20 层几乎不含预测内容，是纯建特征；只有最后 8 层在做预测**。
    所以"学生第 t 轮 = 教师第 t·U 层"对浅臂是个可疑假设：u1t8 的 8 个边界会全落在
    教师的特征构建段（目标读出精度 0.002~0.009）。tail 映射把同样的 8 轮锚到 21~28 层。
    """
    L = int(teacher_layers)
    if explicit:
        return [max(1, min(L, d)) for d in explicit]
    out = []
    for t in range(1, arm_loops + 1):
        d = (t * arm_uniq) if mapping == "head" else (L - (arm_loops - t) * arm_uniq)
        out.append(max(1, min(L, d)))
    return out


def make_insertion_plan(arm_uniq, arm_loops, teacher_layers, args, depth_acc=None):
    """算出插入式蒸馏的监督计划。

    返回 dict：depths=list[T]（每轮的教师对齐层）、hint=list[T]（0/1 权重）、
              logit=list[T]（0/1 是否插输出级监督）
    """
    depths = boundary_depths(arm_uniq, arm_loops, teacher_layers, args.ins_map,
                             explicit=parse_depths(args.ins_depths, arm_loops))
    hint = [args.ins_hint_w] * arm_loops if args.ins_hint_w > 0 else [0.0] * arm_loops
    logit = [0.0] * arm_loops
    if args.ins_logit_w > 0:
        # 只在不低于阈值的深度插**输出级**监督：早层的 lens 读出是垃圾目标，
        # 全权重匹配它 = 教学生"早期输出要烂"。
        tau = args.ins_logit_tau
        for t, d in enumerate(depths):
            ok = True if tau <= 0 else (depth_acc is None or depth_acc[d] >= tau)
            if ok:
                logit[t] = args.ins_logit_w
        # 显存约束：带梯度的逐轮 logits 只留最后 K 轮
        keep = max(0, args.ins_logit_rounds)
        if keep and sum(1 for v in logit if v > 0) > keep:
            live = [t for t in range(arm_loops) if logit[t] > 0][-keep:]
            logit = [logit[t] if t in live else 0.0 for t in range(arm_loops)]
        logit[-1] = 0.0        # 末轮走既有的"真实最终输出"损失，不重复计
    return {"depths": depths, "hint": hint, "logit": logit}


def teacher_depth_targets(teacher, xb, depths, need_logit, temp):
    """教师各对齐深度的目标。

    norm 施加规则（踩过，见 probe_depth.py 自检）：transformers 的 hidden_states
    `[0..L-1]` 是各层**未归一化**输出，`[L]` 已经过 final norm；给 `[L]` 再补一次
    norm 就是重复归一化（实测 0.401 → 0.284）。学生的 h 在循环边界也**未归一化**，
    两者同一约定，可以直接对齐。
    """
    with torch.no_grad():
        out = teacher(xb, output_hidden_states=True)
        hs = out.hidden_states
        L = teacher.config.num_hidden_layers
        t_top1 = out.logits.argmax(-1)
        t_final_prob = F.softmax(out.logits.float() / temp, dim=-1)

        def _read(d):
            return hs[L] if d >= L else teacher.model.norm(hs[d])

        # 逐深度的输出级目标：只在需要时算，算完立刻留 prob（丢 logits）省显存 ——
        # 每层 prob 是 (B,L,151936) 的 fp32，u1t28 全算会吃掉好几 GB。
        t_prob = {d: F.softmax(teacher.lm_head(_read(d)).float() / temp, dim=-1)
                  for d in need_logit}
        # 逐深度的表示级目标：未归一化的那一份（与学生循环边界的 h 同约定）
        hints = {d: (hs[L] if d >= L else hs[d]).detach().clone() for d in set(depths)}
        del out, hs
    return {"hints": hints, "t_prob": t_prob, "t_top1": t_top1, "t_final_prob": t_final_prob}


def relation_matrix(h, eps=1e-6):
    """把 (B,L,D) 的表示压成 (B,L,L) 的**token 间相似度结构**（L2 归一化后的点积）。

    为什么需要这个：hidden 的逐点余弦只约束"每个 token 的方向"，管不到 token 之间的
    结构 —— 而"折叠深度"要迁移的恰恰是**计算模式**（哪些 token 在互相看、信息怎么流），
    不是单个 token 的坐标。相似度矩阵把这一层结构显式暴露出来，代价几乎为零
    （(B,L,L) 而已，教师侧不需要任何额外前向、不需要 eager attention）。

    这是相似性保持蒸馏（RKD / FSP 矩阵那条线）的最小形式。
    """
    z = h.float()
    z = z / (z.norm(dim=-1, keepdim=True) + eps)
    return z @ z.transpose(-1, -2)


def insertion_terms(student_h, plan, targets, grad_logits, temp, hint_kind="cos",
                    rel_w=0.0):
    """算出插入式蒸馏的各附加损失项。全部返回 (标量 tensor 或 None, 统计 dict)。

    **按监督点个数取均值**，不是求和：否则 u1t28（28 个边界）的总 hint 权重会是
    u2t5（5 个边界）的 5.6 倍，`--ins-hint-w` 在不同臂上就不是同一个超参，
    跨臂比较直接作废（踩过）。
    """
    st = {}
    hint_loss, n_hint = None, 0
    for t, (w, d) in enumerate(zip(plan["hint"], plan["depths"])):
        if w <= 0 or t >= len(student_h):
            continue
        a = student_h[t].float()
        b = targets["hints"][d].float()
        if hint_kind == "mse":
            per = F.mse_loss(a, b, reduction="none").mean(-1)
            st["ins_hint_t%d" % (t + 1)] = float(per.mean())
        else:
            # 余弦：尺度无关，对学生早期"量级没长对"更宽容 —— 早期轮次用 MSE
            # 会被量级差异主导，把梯度全花在缩放上。
            per = 1.0 - F.cosine_similarity(a, b, dim=-1)
            st["ins_hint_t%d" % (t + 1)] = float(per.mean())
        term = per.mean() * w
        hint_loss = term if hint_loss is None else hint_loss + term
        n_hint += 1
    if hint_loss is not None and n_hint:
        hint_loss = hint_loss / n_hint

    rel_loss, n_rel = None, 0
    if rel_w > 0:
        for t, (w, d) in enumerate(zip(plan["hint"], plan["depths"])):
            if t >= len(student_h):
                continue
            ws = w if w > 0 else 1.0
            ss = relation_matrix(student_h[t])
            st_t = relation_matrix(targets["hints"][d])
            per = F.mse_loss(ss, st_t, reduction="none").mean(-1)
            st["ins_rel_t%d" % (t + 1)] = float(per.mean())
            term = per.mean() * ws * rel_w
            rel_loss = term if rel_loss is None else rel_loss + term
            n_rel += 1
            del ss, st_t
        if rel_loss is not None and n_rel:
            rel_loss = rel_loss / n_rel

    logit_loss, n_logit = None, 0
    for t, (w, d) in enumerate(zip(plan["logit"], plan["depths"])):
        if w <= 0 or (t + 1) not in grad_logits:
            continue
        s_log = F.log_softmax(grad_logits[t + 1].float() / temp, dim=-1)
        tp = targets["t_prob"][d]
        kl = F.kl_div(s_log, tp, reduction="none").sum(-1)
        term = (kl * (temp ** 2)).mean() * w
        logit_loss = term if logit_loss is None else logit_loss + term
        n_logit += 1
        with torch.no_grad():
            st["ins_logit_agree_t%d" % (t + 1)] = float(
                (grad_logits[t + 1].argmax(-1) == tp.argmax(-1)).float().mean())
        del s_log, kl
    if logit_loss is not None and n_logit:
        logit_loss = logit_loss / n_logit
    return hint_loss, rel_loss, logit_loss, st


# ---------------------------------------------------------------- 训练

def distillation_step(student, teacher, xb, yb, temp, alpha_ce, device,
                      ins_plan=None, hint_kind="cos", want_stats=True, ins_rel_w=0.0):
    """返回 (loss, 统计)。教师概率算完即释放 logits，只留 probs（省显存）。

    ins_plan 不为 None 时启用**插入式蒸馏**：在每轮循环边界插表示(hint)监督、
    在末几轮插输出(logit)监督。分子项权重见 make_insertion_plan。
    """
    grad_rounds = None
    if ins_plan is not None:
        grad_rounds = [t + 1 for t, w in enumerate(ins_plan["logit"]) if w > 0]
    need_logit = sorted({ins_plan["depths"][t] for t, w in enumerate(ins_plan["logit"])
                         if w > 0}) if ins_plan is not None else []

    if ins_plan is not None:
        tg = teacher_depth_targets(teacher, xb, ins_plan["depths"], need_logit, temp)
        t_prob, t_top1, t_final_prob = tg["t_prob"], tg["t_top1"], tg["t_final_prob"]
    else:
        tg = None
        with torch.no_grad():
            t_logits = teacher(xb).logits
            t_prob = F.softmax(t_logits.float() / temp, dim=-1)
            t_top1 = t_logits.argmax(-1)
            del t_logits

    if ins_plan is not None:
        s_logits, aux = student(xb, collect_hidden=True, grad_rounds=grad_rounds)
        aux = aux or {}
    else:
        s_logits, aux = student(xb)

    s_log = F.log_softmax(s_logits.float() / temp, dim=-1)
    kl = F.kl_div(s_log, t_final_prob if ins_plan is not None else t_prob,
                  reduction="none").sum(-1)
    loss = (kl * (temp ** 2)).mean()
    ce = F.cross_entropy(s_logits.float().view(-1, s_logits.size(-1)), yb.view(-1))
    loss = loss + alpha_ce * ce

    st = {}
    if ins_plan is not None:
        h_loss, r_loss, l_loss, ist = insertion_terms(
            aux.get("round_hidden") or [], ins_plan, tg,
            aux.get("round_logits_grad") or {}, temp, hint_kind, rel_w=ins_rel_w)
        st.update(ist)
        if h_loss is not None:
            loss = loss + h_loss
        if r_loss is not None:
            loss = loss + r_loss
        if l_loss is not None:
            loss = loss + l_loss
        del t_prob
        # 循环边界的表示对齐度（余弦均值）—— 看监督有没有真的把表示推过去
        with torch.no_grad():
            hh = aux.get("round_hidden") or []
            if hh:
                cs = []
                for t, d in enumerate(ins_plan["depths"]):
                    a, b = hh[t].float(), tg["hints"][d].float()
                    cs.append(float(F.cosine_similarity(a, b, dim=-1).mean()))
                st["ins_hint_cos_mean"] = sum(cs) / len(cs)
                st["ins_hint_cos_first"] = cs[0]
                st["ins_hint_cos_last"] = cs[-1]
    else:
        del t_prob

    with torch.no_grad():
        agree = (s_logits.argmax(-1) == t_top1).float().mean()
        acc = (s_logits.argmax(-1) == yb).float().mean()
        kl_scalar = (kl / xb.numel()).mean() if xb.numel() else kl.mean()
    st.update({"kl": float(kl_scalar), "agree_top1": float(agree), "student_acc": float(acc)})
    return loss, st


@torch.no_grad()
def teacher_acc(teacher, x, y, device, max_batches=0, cap=32):
    """教师自己在给定窗口上的 top1 准确率 —— **蒸馏上限的参照线**。

    踩过的坑（口径不可比）：早先这里默认 `max_batches=4` 且 batch=1，而学生评测走的是
    `evaluate(max_batches=8)`，**两边用的窗口子集不同**，于是"上限"与"学生"根本不在同一
    批输入上 —— 同一个小说留出集，教师上限被这个口径测出过 0.3252 / 0.4014 / 0.4287
    三个值（4/12/24 个窗口）。那个"差距=上限-学生"列因此不可信。
    现在默认 `max_batches=0` = 尽量用满留出集（上限 cap），学生侧也走同一个数。
    """
    teacher.eval()
    n_all = x.size(0)
    nb = n_all if max_batches <= 0 else min(max_batches, n_all)
    nb = min(nb, cap)
    if nb <= 0:
        return 0.0, 0
    hit = torch.zeros((), device=device)
    tot = 0
    for i in range(nb):
        lg = teacher(x[i:i + 1]).logits
        hit = hit + (lg.argmax(-1) == y[i:i + 1]).float().sum()
        tot += y[i:i + 1].numel()
        del lg
    return float(hit / max(tot, 1)), nb


@torch.no_grad()
def fit_verdict(t_train, s_train, t_hold, s_hold):
    """判据：欠拟合还是过拟合 —— 这决定"数据不够"这个解释成不成立。

    过拟合（训练高、留出低）→ 有可能是数据量问题；欠拟合（两边都低、都远离教师）
    → **一定不是数据量问题**（欠拟合加数据没用，只能改函数类/目标/优化）。
    """
    tg = t_train - s_train
    hg = t_hold - s_hold
    if t_train <= 0 or t_hold <= 0:
        return "无教师参照，无法判定"
    if hg > max(tg, 0.02) * 1.6:
        return "过拟合迹象（留出差距 %.3f >> 训练差距 %.3f）→ 可能是数据量问题" % (hg, tg)
    if tg > 0.25 and hg > 0.25:
        return ("欠拟合（训练差距 %.3f、留出差距 %.3f，两边都远离教师）→ "
                "**不是数据量问题**：加数据治不好，要改可学函数类/目标/初始化" % (tg, hg))
    return "介于两者之间（训练差距 %.3f、留出差距 %.3f）" % (tg, hg)


@torch.no_grad()
def evaluate(student, teacher, x, y, temp, device, max_batches=0, collect_rounds=True, cap=32):
    """max_batches<=0 = 用满留出集（上限 cap 个窗口）。必须与 teacher_acc 用同一个数，
    否则"上限 - 学生"这个差距是拿两批不同输入相减，没有意义。"""
    student.eval()
    agg = {"kl": 0.0, "agree_top1": 0.0, "student_acc": 0.0}
    n = 0
    round_acc = None
    nb = x.size(0) if max_batches <= 0 else min(max_batches, x.size(0))
    nb = min(nb, cap)
    for i in range(nb):
        xb, yb = x[i:i + 1], y[i:i + 1]
        t_logits = teacher(xb).logits
        t_prob = F.softmax(t_logits.float() / temp, dim=-1)
        t_top1 = t_logits.argmax(-1)
        del t_logits
        s_logits, aux = student(xb, collect_rounds=collect_rounds)
        rounds = (aux or {}).get("round_logits")
        s_log = F.log_softmax(s_logits.float() / temp, dim=-1)
        agg["kl"] += float(F.kl_div(s_log, t_prob, reduction="batchmean"))
        agg["agree_top1"] += float((s_logits.argmax(-1) == t_top1).float().mean())
        agg["student_acc"] += float((s_logits.argmax(-1) == yb).float().mean())
        if rounds:
            if round_acc is None:
                round_acc = [0.0] * len(rounds)
            for t, rl in enumerate(rounds):
                round_acc[t] += float((rl.argmax(-1) == t_top1).float().mean())
        del t_prob
        n += 1
    for k in agg:
        agg[k] /= max(n, 1)
    if round_acc:
        agg["round_agree"] = [v / max(n, 1) for v in round_acc]
    student.train()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "WEASEL_LLM_MODEL", r"E:\DSH_data\研究\models\Qwen3-0.6B-Base"),
        help="教师（同时决定学生的 config 与词表）")
    ap.add_argument("--arm", default="u4t7", choices=sorted(ARMS))
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "scratch"],
                    help="pretrained=学生 U 个共享层吃教师前 U 层（检验可迁移性）；scratch=随机初始化")
    ap.add_argument("--round-emb-init", default="zeros", choices=["zeros", "normal"])
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4, help="每步总 batch（含梯度累积）")
    ap.add_argument("--micro", type=int, default=1, help="显存不够就调小到 1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--alpha-ce", type=float, default=0.5, help="硬标签 CE 的权重")
    ap.add_argument("--eval-batches", type=int, default=16,
                    help="评测/教师上限用的窗口数（<=0 表示用满留出集）。"
                         "**教师上限与学生必须同一个数**，否则差距列不可信")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--novel", default=os.environ.get(
        "DISTILL_NOVEL", "src/银砂纪年 第一卷.txt"))
    ap.add_argument("--typing", default=os.environ.get("DISTILL_TYPING", "src/typing.txt"))
    ap.add_argument("--mix-typing", type=float, default=0.3,
                    help="打字语料在训练流里的字符占比；0=只用小说")
    ap.add_argument("--holdout-typing", type=float, default=0.1,
                    help="打字语料尾部留出比例（打字域评测集）")
    ap.add_argument("--out", default="runs/distill")
    ap.add_argument("--train-scope", default="head",
                    choices=["head", "last1", "last2", "last4", "last8", "layers", "all"],
                    help="head=只训头/嵌入（线上纪律）；lastN=额外解冻末 N 个唯一层；"
                         "layers=全部层；all=全部参数")
    ap.add_argument("--lr-backbone", type=float, default=0.0,
                    help="主干学习率；0 表示用 --lr/10")
    ap.add_argument("--resume", default="", help="从已有 student.pt 继续训（两步法：先上课再冻结训头）")
    # —— 插入式蒸馏（在每轮循环边界插监督信号）——
    ap.add_argument("--ins-hint-w", type=float, default=0.0,
                    help="循环边界表示(hint)监督的权重；0=关。这是插入式蒸馏的主项")
    ap.add_argument("--ins-hint-kind", default="cos", choices=["cos", "mse"],
                    help="表示对齐用余弦（尺度无关，早期轮次推荐）还是 MSE")
    ap.add_argument("--ins-logit-w", type=float, default=0.0,
                    help="循环边界输出(logit)监督的权重；0=关。早层 lens 读出是垃圾目标，慎用")
    ap.add_argument("--ins-logit-tau", type=float, default=0.10,
                    help="只对教师同深度 lens 精度 >= tau 的边界插输出级监督（0=全插）")
    ap.add_argument("--ins-logit-rounds", type=int, default=4,
                    help="带梯度的逐轮 logits 最多留最后 K 轮（显存约束）")
    ap.add_argument("--ins-map", default="head", choices=["head", "tail"],
                    help="深度对齐映射：head=t·U（从头）；tail=L-(T-t)·U（从尾，锚在教师精修段）")
    ap.add_argument("--ins-depths", default="",
                    help="显式边界深度表（逗号分隔，个数必须等于轮数）—— 非均匀折叠："
                         "教师计算密度不均（前 20 层建特征、后 8 层做预测），"
                         "浅层应大步、深层应小步，如 \"4,8,12,16,20,22,24,26,28\"")
    ap.add_argument("--ins-rel-w", type=float, default=0.0,
                    help="循环边界**关系(相似度结构)**监督的权重；0=关。"
                         "逐点余弦管不到 token 之间的结构，而折叠要迁移的正是计算模式")
    ap.add_argument("--smoke", action="store_true", help="只抽 200 条 / 跑 20 步 / 跳过留出评估")
    ap.add_argument("--teacher-baseline", action="store_true", default=True,
                    help="打印/记录教师自身在留出集上的 top1（蒸馏上限参照）")
    ap.add_argument("--no-teacher-baseline", dest="teacher_baseline", action="store_false")
    ap.add_argument("--seed", type=int, default=20260913)
    args = ap.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 20)
        args.eval_every = min(args.eval_every, 10)
        args.batch = min(args.batch, 2)
        args.micro = min(args.micro, 1)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    log("device=%s  arm=%s  init=%s" % (device, args.arm, args.init))

    tok = AutoTokenizer.from_pretrained(args.model)
    tr_text, va_text, va_typing = build_corpus(args)
    x, y = make_windows(tok, tr_text, args.seq, device)
    xv, yv = make_windows(tok, va_text, args.seq, device)
    xt = yt = None
    if va_typing:
        xt, yt = make_windows(tok, va_typing, args.seq, device)
        log("打字域留出窗口 %d 个" % xt.size(0))
    log("训练窗口 %d 个（%d token）｜小说留出窗口 %d 个" % (x.size(0), x.numel(), xv.size(0)))
    if x.size(0) < 4:
        raise SystemExit("训练窗口太少，检查语料与 --seq")

    log("加载教师（fp16，冻结）…")
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    uniq, loops = ARMS[args.arm]
    student = build_student(args.model, uniq, loops, args.round_emb_init).to(device)
    rep = student.param_report()
    log("学生 U=%d T=%d 等效深度=%d" % (uniq, loops, student.effective_depth))
    log("参数：blocks=%s  embed=%s  lm_head=%s  tie=%s  唯一合计=%s"
        % (f"{rep['blocks']:,}", f"{rep['embed']:,}", f"{rep['lm_head']:,}",
           rep["tied"], f"{rep['total_unique']:,}"))
    t_total = sum(v.numel() for v in teacher.state_dict().values())
    t_params = sum(p.numel() for p in teacher.parameters())
    log("教师 state_dict %s ／ parameters() %s（tie 时二者不等，属正常）"
        % (f"{t_total:,}", f"{t_params:,}"))
    log("学生唯一参数 %s  学生/教师(state_dict) = %.3f"
        % (f"{rep['total_unique']:,}", rep["total_unique"] / t_total))

    # ---- 插入式蒸馏计划：把监督点铺到循环边界上 ----
    ins_plan = None
    if args.ins_hint_w > 0 or args.ins_logit_w > 0 or args.ins_rel_w > 0:
        depth_acc = None
        if args.ins_logit_w > 0 and args.ins_logit_tau > 0:
            p = os.path.join(HERE, "probe_depth_result.json")
            if os.path.exists(p):
                try:
                    dj = json.load(open(p, encoding="utf-8"))["per_layer_acc"]
                    key = "typing" if dj.get("typing") else "novel"
                    depth_acc = dj[key]
                    log("载入逐层 lens 曲线（%s，%d 层）用于 tau=%.2f 过滤"
                        % (key, len(depth_acc) - 1, args.ins_logit_tau))
                except Exception as e:
                    log("逐层曲线载入失败（%s）→ tau 过滤退化为全插" % e)
        ins_plan = make_insertion_plan(uniq, loops, teacher.config.num_hidden_layers,
                                       args, depth_acc)
        log("插入式蒸馏：map=%s 边界深度=%s"
            % ("explicit" if args.ins_depths else args.ins_map, ins_plan["depths"]))
        log("  hint 权重 %s ｜ 关系 权重 %.2f ｜ logit 权重 %s（tau=%.2f，最多带梯度 %d 轮）"
            % (ins_plan["hint"], args.ins_rel_w, ins_plan["logit"],
               args.ins_logit_tau, args.ins_logit_rounds))
        if not any(ins_plan["hint"]) and not any(ins_plan["logit"]) and args.ins_rel_w <= 0:
            raise SystemExit("插入式蒸馏开了但所有监督点权重为 0，检查 --ins-* 参数")
        if args.ins_logit_w > 0 and not any(ins_plan["logit"]):
            # 静默无监督是最糟的失败模式：命令看着成功、其实没插任何输出级监督。
            log("警告：--ins-logit-w=%.2f 但 tau=%.2f 把全部边界都过滤掉了"
                "（这些深度的教师读出精度都不够），等效于没插 logit 监督"
                % (args.ins_logit_w, args.ins_logit_tau))

    if args.init == "pretrained":
        used, missing = student.init_from_teacher(teacher.state_dict())
        student.tie_follows()          # 重新绑定，防止 copy_ 把 tie 弄丢
        log("从教师初始化：%d 个张量载入，%d 个未命中" % (len(used), len(missing)))
        if missing:
            log("未命中：%s" % missing[:8])

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        miss = student.load_state_dict(sd, strict=False)
        student.tie_follows()
        log("续训载入 %s（缺失 %d / 多余 %d）"
            % (args.resume, len(miss.missing_keys), len(miss.unexpected_keys)))

    # 冻结策略：默认冻主干只训头（与线上在线 RL 同一纪律）。
    # 但"随机主干 + 只训头"只能学出上下文无关的输出偏置，学不出表示 ——
    # 所以留 --train-scope 用来把主干也解开，检验表示到底能不能学出来。
    sc = args.train_scope
    n_unfreeze = {"head": 0, "last1": 1, "last2": 2, "last4": 4, "last8": 8}.get(sc, uniq)
    for p in student.parameters():
        p.requires_grad = False
    head_p = student.embed_tokens.weight if rep["tied"] else student.lm_head.weight
    head_p.requires_grad = True
    groups = [{"params": [head_p], "lr": args.lr}]
    bb = []
    if sc in ("head",):
        pass
    elif sc == "all":
        for n, p in student.named_parameters():
            if p is head_p:
                continue
            p.requires_grad = True
            bb.append(p)
    else:
        for li in range(max(0, uniq - n_unfreeze), uniq):
            for p in student.layers[li].parameters():
                p.requires_grad = True
                bb.append(p)
        if sc == "layers":
            for li in range(0, max(0, uniq - n_unfreeze)):
                for p in student.layers[li].parameters():
                    p.requires_grad = True
                    bb.append(p)
        for p in student.norm.parameters():
            p.requires_grad = True
            bb.append(p)
    if bb:
        groups.append({"params": bb, "lr": args.lr_backbone or args.lr / 10})
    trainable = [p for g in groups for p in g["params"]]
    n_train = sum(p.numel() for p in trainable)
    n_bb = sum(p.numel() for p in bb)
    log("可训参数 = %s（主干 %s，scope=%s%s）"
        % (f"{n_train:,}", f"{n_bb:,}", sc,
           "" if not bb else "，主干 lr=%.1e" % (args.lr_backbone or args.lr / 10)))

    # 阈值参照：先算教师自己在留出集上的 top1（smoke 路径跳过，保持快）
    # **口径必须与学生评测一致**：教师上限与学生用同一批窗口，否则"差距"列不可信。
    t_novel = t_typing = t_train = None
    if not args.smoke:
        ev_b = args.eval_batches
        if args.teacher_baseline:
            t_novel, nb_n = teacher_acc(teacher, xv, yv, device, max_batches=ev_b)
            log("[上限·小说留出] 教师自身 top1 = %.4f（%d 窗口，与学生同批）" % (t_novel, nb_n))
            if xt is not None:
                t_typing, nb_t = teacher_acc(teacher, xt, yt, device, max_batches=ev_b)
                log("[上限·打字留出] 教师自身 top1 = %.4f（%d 窗口）" % (t_typing, nb_t))
            # 训练窗口上的教师上限：没有这个数就分不清"欠拟合"和"过拟合"，
            # 而这两者的解法完全不同（欠拟合加数据没用）。
            t_train, nb_tr = teacher_acc(teacher, x, y, device, max_batches=ev_b)
            log("[上限·训练窗口] 教师自身 top1 = %.4f（%d 窗口，抽样）" % (t_train, nb_tr))
        ev0 = evaluate(student, teacher, xv, yv, args.temp, device, max_batches=ev_b)
        log("[基线·小说留出] kl=%.4f agree_top1=%.4f student_acc=%.4f%s"
            % (ev0["kl"], ev0["agree_top1"], ev0["student_acc"],
               "" if t_novel is None else "（教师 %.4f）" % t_novel))
        if t_train is not None:
            s_tr = evaluate(student, teacher, x, y, args.temp, device,
                            max_batches=ev_b, collect_rounds=False)
            log("[基线·判定] 训练窗口学生 acc=%.4f（教师 %.4f）→ %s"
                % (s_tr["student_acc"], t_train,
                   fit_verdict(t_train, s_tr["student_acc"], t_novel, ev0["student_acc"])))
        if xt is not None:
            et0 = evaluate(student, teacher, xt, yt, args.temp, device, max_batches=ev_b)
            log("[基线·打字留出] kl=%.4f agree_top1=%.4f student_acc=%.4f%s"
                % (et0["kl"], et0["agree_top1"], et0["student_acc"],
                   "" if t_typing is None else "（教师 %.4f）" % t_typing))

    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)

    # 线性预热 + 余弦退火。不用 OneCycleLR：它的预热由 pct_start 决定，
    # 步数很少时（-smoke）会除零崩溃；LambdaLR 没有这个边角。
    warm = max(1, min(args.warmup, max(1, args.steps // 10)))
    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, args.steps - warm)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    metrics_path = os.path.join(args.out, "metrics.jsonl")
    meta = vars(args).copy()
    meta.update({"uniq": uniq, "loops": loops, "effective_depth": student.effective_depth,
                 "student_params": rep, "teacher_params": t_total,
                 "ins_plan": ins_plan,          # 插入式蒸馏的完整计划（含每个边界对齐的教师深度）
                 "train_scope": args.train_scope,
                 "train_windows": int(x.size(0)), "holdout_windows": int(xv.size(0)),
                 "teacher_acc_novel": t_novel, "teacher_acc_typing": t_typing,
                 "teacher_acc_train": t_train})
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    micro_per_step = max(1, args.batch // args.micro)
    # smoke 路径不会走进上面的 baseline 分支，这里兜底，保证 meta/日志引用不炸
    if "t_novel" not in dir():
        t_novel = t_typing = None
    ptr = 0
    student.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        acc_stats = {}
        for _ in range(micro_per_step):
            if ptr + args.micro > x.size(0):
                ptr = 0
            xb = x[ptr:ptr + args.micro]
            yb = y[ptr:ptr + args.micro]
            ptr += args.micro
            loss, st = distillation_step(student, teacher, xb, yb, args.temp,
                                         args.alpha_ce, device,
                                         ins_plan=ins_plan, hint_kind=args.ins_hint_kind,
                                         ins_rel_w=args.ins_rel_w)
            (loss / micro_per_step).backward()
            for k, v in st.items():
                acc_stats[k] = acc_stats.get(k, 0.0) + v / micro_per_step
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()
        sched.step()

        if step % 20 == 0 or step == 1:
            extra = ""
            if ins_plan is not None and "ins_hint_cos_mean" in acc_stats:
                extra = (" ins_cos(首/末/均)=%.3f/%.3f/%.3f"
                         % (acc_stats.get("ins_hint_cos_first", 0.0),
                            acc_stats.get("ins_hint_cos_last", 0.0),
                            acc_stats.get("ins_hint_cos_mean", 0.0)))
            log("step %d/%d loss=%.4f kl=%.4f agree=%.4f acc=%.4f lr=%.2e %.1fs%s"
                % (step, args.steps, float(loss.detach()), acc_stats["kl"], acc_stats["agree_top1"],
                   acc_stats["student_acc"], sched.get_last_lr()[0], time.time() - t0, extra))

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(student, teacher, xv, yv, args.temp, device, max_batches=args.eval_batches,
                          collect_rounds=(step == args.steps or step % (args.eval_every * 5) == 0))
            rec = {"step": step, **acc_stats, **{k: v for k, v in ev.items() if k != "round_agree"}}
            rec["teacher_acc_novel"] = t_novel       # 上限参照，随每行落盘，绘图时可直接画平行线
            rec["teacher_acc_train"] = t_train
            if "round_agree" in ev:
                ra = ev["round_agree"]
                rec["round_agree"] = ra
                rec["round_agree_head"] = ra[:4]
                log("  早退曲线（第1..%d轮 与教师top1一致率）：%s"
                    % (len(ra), " ".join("%.3f" % v for v in ra[:8])))
            if xt is not None:
                # 打字域也收早退曲线：插入式蒸馏的**产品收益**就在这条线上 ——
                # 第 t 轮就能用，意味着在线推理只需跑 t 轮（省算力）。
                et = evaluate(student, teacher, xt, yt, args.temp, device,
                              max_batches=args.eval_batches,
                              collect_rounds=(step == args.steps))
                rec["typing_kl"] = et["kl"]
                rec["typing_agree_top1"] = et["agree_top1"]
                rec["typing_student_acc"] = et["student_acc"]
                rec["teacher_acc_typing"] = t_typing
                if "round_agree" in et:
                    rec["typing_round_agree"] = et["round_agree"]
                    log("  打字早退曲线（第1..%d轮）：%s"
                        % (len(et["round_agree"]),
                           " ".join("%.3f" % v for v in et["round_agree"][:8])))
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log("  [eval·小说] step=%d kl=%.4f agree_top1=%.4f student_acc=%.4f%s"
                % (step, ev["kl"], ev["agree_top1"], ev["student_acc"],
                   "" if t_novel is None else "  上限(教师)=%.4f  差距=%.4f"
                   % (t_novel, t_novel - ev["student_acc"])))
            if xt is not None:
                log("  [eval·打字] step=%d kl=%.4f agree_top1=%.4f student_acc=%.4f%s"
                    % (step, et["kl"], et["agree_top1"], et["student_acc"],
                       "" if t_typing is None else "  上限(教师)=%.4f  差距=%.4f"
                       % (t_typing, t_typing - et["student_acc"])))
            # 判定行：欠拟合还是过拟合 —— 直接决定"数据不够"这个解释成不成立
            if t_train is not None:
                s_tr = evaluate(student, teacher, x, y, args.temp, device,
                                max_batches=args.eval_batches, collect_rounds=False)
                rec["train_student_acc"] = s_tr["student_acc"]
                log("  [判定] 训练acc=%.4f（教师 %.4f，差距 %.3f）｜ 留出acc=%.4f（差距 %.3f）→ %s"
                    % (s_tr["student_acc"], t_train, t_train - s_tr["student_acc"],
                       ev["student_acc"], (t_novel or 0) - ev["student_acc"],
                       fit_verdict(t_train, s_tr["student_acc"], t_novel, ev["student_acc"])))

    ckpt = os.path.join(args.out, "student.pt")
    torch.save({"state_dict": student.state_dict(), "meta": meta}, ckpt)
    log("已保存 %s" % ckpt)
    log("完成。metrics: %s" % metrics_path)


if __name__ == "__main__":
    main()
