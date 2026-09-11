#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对照实验：把「损失打在谁身上」和「排名权重方向」拆开，看改进来自哪一个。

    A 候选+正挂   损失打在模型自己提的候选上（字符部分匹配，旧规则），
                  排名越靠前权重越大（旧阶梯）
    B 候选+倒挂   同上，但排名越靠后权重越大
    C 真值+正挂   损失打在真正出现的下一个 token 上（交叉熵），正挂阶梯
    D 真值+倒挂   同上，倒挂阶梯（当前线上的规则）
    E 真值+无权重 交叉熵，权重恒为 1（没有阶梯）

只变这两件事，其余全部固定：同一段 1000 字、同一起点（线上头）、同一个
lr 5e-4、同一个普通 SGD、同样 chunk=400/prompt=64、标点同样不计分、
表外权重同样取 1.0（候选臂因为无法给"没有匹配"打分，那一支仍然是零梯度）。

全程在显存里跑，不落任何权重文件；测完把权重还原成训练前的样子。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "tools", "LlamaTreeExp")
sys.path.insert(0, EXP)

import torch
import probe_eval as pe
import unified_pipeline as up

NOVEL = r"C:\Users\Feng\Desktop\共享文件夹\银砂纪年\银砂纪年 第一卷 少女们，学院，龙与迫近的危机.txt"
START, LEN = 24319, 1000
CHUNK, PROMPT, K, LR = 400, 64, 20, 5e-4
CKPT = os.path.join(EXP, "diag", "checkpoints_online", "lm_head_t0.pt")


def ladder(rank, k, kind):
    if rank < 0:
        return 1.0                      # 表外：所有臂都用满权重
    if kind == "desc":
        return (k - rank) / float(k)    # 正挂：第1名 100% … 第20名 5%
    if kind == "asc":
        return (rank + 1) / float(k)    # 倒挂：第1名 5% … 第20名 100%
    return 1.0                          # 无权重


def train_arm(engine, text, target, weight, steps_log):
    """target: 'cand' | 'true';  weight: 'desc' | 'asc' | 'none'"""
    tok, dev = engine.tokenizer, engine.device
    W = engine.model.lm_head.weight
    steps = 0
    pos = 0
    while pos < len(text):
        head = text[max(0, pos - PROMPT):pos]
        if not head:
            pos += 1
            continue
        seg = text[pos:pos + CHUNK]
        ids = tok.encode(head + seg, add_special_tokens=False)
        n_head = len(tok.encode(head, add_special_tokens=False))
        pos += len(seg)
        if len(ids) <= n_head:
            continue
        pieces = [tok.decode([i]) for i in ids]
        offs = [0] * (len(pieces) + 1)
        for i, p in enumerate(pieces):
            offs[i + 1] = offs[i] + len(p)
        with torch.no_grad():
            out = engine.model.model(input_ids=torch.tensor([ids], device=dev),
                                     use_cache=True)
        engine.model.train()
        for row in range(n_head, len(ids)):
            tid = ids[row]
            if up.is_no_target(tok.decode([tid])):
                continue
            hidden = out.last_hidden_state[0, row - 1, :]
            logits = engine.model.lm_head(
                hidden.to(torch.float32).unsqueeze(0))[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            best, best_s = None, 0.0
            with torch.no_grad():
                probs = log_probs.detach().exp()
                top_p, top_id = torch.topk(probs, K)
                if target == "true":
                    m = (top_id == tid).nonzero(as_tuple=True)[0]
                    rank = int(m[0].item()) if m.numel() else -1
                else:
                    rest = "".join(pieces[row:])
                    for r, cand in enumerate(top_id.tolist()):
                        ct = tok.decode([int(cand)])
                        if not ct:
                            continue
                        n = 0
                        while (n < len(ct) and n < len(rest)
                               and ct[n] == rest[n]):
                            n += 1
                        if not n:
                            continue
                        s = ladder(r, K, weight) * (float(n) / float(len(ct)))
                        if s > best_s:
                            best, best_s = int(cand), s
            # 权重算完之后再建图：放在 no_grad 里就没有 grad_fn 了
            if target == "true":
                loss = -ladder(rank, K, weight) * log_probs[tid]
            else:
                if best is None:
                    continue              # 没有匹配候选：零梯度
                loss = -best_s * log_probs[best]
            engine.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            engine.optimizer.step()
            steps += 1
        engine.model.eval()
        engine.optimizer.zero_grad(set_to_none=True)
    steps_log.append(steps)


def main():
    raw = pe.read_text(NOVEL)
    train_text = raw[START:START + LEN]
    slices = [("前1000字(训练)", train_text),
              ("后1000字(留出)", raw[START + LEN:START + 2 * LEN])]

    engine = up.TreeEngine(up.MODEL_PATH, lr=LR, device=up.DEVICE, dtype="float16")
    W = engine.model.lm_head.weight
    engine.load_checkpoint(CKPT)
    base = W.detach().clone()

    arms = [("A 候选+正挂", "cand", "desc"),
            ("B 候选+倒挂", "cand", "asc"),
            ("C 真值+正挂", "true", "desc"),
            ("D 真值+倒挂", "true", "asc"),
            ("E 真值+无权重", "true", "none")]

    print("\n%-14s %-16s %-10s %-8s %-8s %-7s %-7s" %
          ("臂", "切片", "前20累计p", "命中率", "脱靶率", "最长链", "top1"))
    rows = []

    def measure(tag, name, text):
        r = pe.walk_metrics(engine, text)
        print("%-14s %-16s %-10.4f %-8.1f %-8.1f %-7d %-7.1f"
              % (tag, name, r["mass20"], r["hit_rate"], r["miss_rate"],
                 r["longest"], r["hit1"]), flush=True)
        rows.append((tag, name, r))
        return r

    for name, text in slices:
        measure("训练前", name, text)
    print()

    for tag, target, weight in arms:
        with torch.no_grad():
            W.copy_(base)
        steps = []
        t0 = time.time()
        train_arm(engine, train_text, target, weight, steps)
        for name, text in slices:
            measure(tag, name, text)
        print("    (%s 训练 %d 步 / %.0f 秒)" % (tag, steps[0], time.time() - t0),
              flush=True)
        print()

    with torch.no_grad():
        W.copy_(base)
    print("权重已还原为训练前的状态（全程没有写盘）")


if __name__ == "__main__":
    main()
