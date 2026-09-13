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


# ---------------------------------------------------------------- 训练

def distillation_step(student, teacher, xb, yb, temp, alpha_ce, device):
    """返回 (loss, 统计)。教师概率算完即释放 logits，只留 probs（省显存）。"""
    with torch.no_grad():
        t_logits = teacher(xb).logits
        t_prob = F.softmax(t_logits.float() / temp, dim=-1)
        t_top1 = t_logits.argmax(-1)
        del t_logits
    s_logits = student(xb)[0]
    s_log = F.log_softmax(s_logits.float() / temp, dim=-1)
    kl = F.kl_div(s_log, t_prob, reduction="none").sum(-1)
    del t_prob
    loss = (kl * (temp ** 2)).mean()
    ce = F.cross_entropy(s_logits.float().view(-1, s_logits.size(-1)), yb.view(-1))
    loss = loss + alpha_ce * ce
    with torch.no_grad():
        agree = (s_logits.argmax(-1) == t_top1).float().mean()
        acc = (s_logits.argmax(-1) == yb).float().mean()
        kl_scalar = (kl / xb.numel()).mean() if xb.numel() else kl.mean()
    return loss, {"kl": float(kl_scalar), "agree_top1": float(agree), "student_acc": float(acc)}


@torch.no_grad()
def teacher_acc(teacher, x, y, device, max_batches=4):
    """教师自己在留出集上的 top1 准确率 —— **蒸馏上限的参照线**。

    没有这个数，学生的 student_acc 无法解读：0.38 到底是"学生差"还是"这个任务本身就
    只有 0.45 可拿"，只有教师自己打出的分才说得清。
    """
    teacher.eval()
    acc = 0.0
    nb = min(max_batches, x.size(0))
    for i in range(nb):
        lg = teacher(x[i:i + 1]).logits
        acc += float((lg.argmax(-1) == y[i:i + 1]).float().mean())
    return acc / max(nb, 1)


@torch.no_grad()
def evaluate(student, teacher, x, y, temp, device, max_batches=8, collect_rounds=True):
    student.eval()
    agg = {"kl": 0.0, "agree_top1": 0.0, "student_acc": 0.0}
    n = 0
    round_acc = None
    nb = min(max_batches, x.size(0))
    for i in range(nb):
        xb, yb = x[i:i + 1], y[i:i + 1]
        t_logits = teacher(xb).logits
        t_prob = F.softmax(t_logits.float() / temp, dim=-1)
        t_top1 = t_logits.argmax(-1)
        del t_logits
        s_logits, rounds = student(xb, collect_rounds=collect_rounds)
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
    t_novel = t_typing = None
    if not args.smoke:
        if args.teacher_baseline:
            t_novel = teacher_acc(teacher, xv, yv, device, max_batches=4)
            log("[上限·小说留出] 教师自身 top1 = %.4f" % t_novel)
            if xt is not None:
                t_typing = teacher_acc(teacher, xt, yt, device, max_batches=4)
                log("[上限·打字留出] 教师自身 top1 = %.4f" % t_typing)
        ev0 = evaluate(student, teacher, xv, yv, args.temp, device, max_batches=4)
        log("[基线·小说留出] kl=%.4f agree_top1=%.4f student_acc=%.4f%s"
            % (ev0["kl"], ev0["agree_top1"], ev0["student_acc"],
               "" if t_novel is None else "（教师 %.4f）" % t_novel))
        if xt is not None:
            et0 = evaluate(student, teacher, xt, yt, args.temp, device, max_batches=4)
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
                 "train_windows": int(x.size(0)), "holdout_windows": int(xv.size(0)),
                 "teacher_acc_novel": t_novel, "teacher_acc_typing": t_typing})
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
                                         args.alpha_ce, device)
            (loss / micro_per_step).backward()
            for k, v in st.items():
                acc_stats[k] = acc_stats.get(k, 0.0) + v / micro_per_step
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()
        sched.step()

        if step % 20 == 0 or step == 1:
            log("step %d/%d loss=%.4f kl=%.4f agree=%.4f acc=%.4f lr=%.2e %.1fs"
                % (step, args.steps, float(loss.detach()), acc_stats["kl"], acc_stats["agree_top1"],
                   acc_stats["student_acc"], sched.get_last_lr()[0], time.time() - t0))

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(student, teacher, xv, yv, args.temp, device,
                          collect_rounds=(step == args.steps or step % (args.eval_every * 5) == 0))
            rec = {"step": step, **acc_stats, **{k: v for k, v in ev.items() if k != "round_agree"}}
            rec["teacher_acc_novel"] = t_novel       # 上限参照，随每行落盘，绘图时可直接画平行线
            if "round_agree" in ev:
                ra = ev["round_agree"]
                rec["round_agree"] = ra
                rec["round_agree_head"] = ra[:4]
                log("  早退曲线（第1..%d轮 与教师top1一致率）：%s"
                    % (len(ra), " ".join("%.3f" % v for v in ra[:8])))
            if xt is not None:
                et = evaluate(student, teacher, xt, yt, args.temp, device, max_batches=4)
                rec["typing_kl"] = et["kl"]
                rec["typing_agree_top1"] = et["agree_top1"]
                rec["typing_student_acc"] = et["student_acc"]
                rec["teacher_acc_typing"] = t_typing
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

    ckpt = os.path.join(args.out, "student.pt")
    torch.save({"state_dict": student.state_dict(), "meta": meta}, ckpt)
    log("已保存 %s" % ckpt)
    log("完成。metrics: %s" % metrics_path)


if __name__ == "__main__":
    main()
