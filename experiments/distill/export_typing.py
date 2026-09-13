# -*- coding: utf-8 -*-
"""把加密的打字语料解密出来，落成纯文本，供蒸馏训练用。

segments.jsonl / corpus.jsonl 是 AES-256-GCM（corpus.py），读出来是 base64 密文；
训练侧不需要关心这些，所以这里只做一次解密导出：

    python export_typing.py                     # 默认读 tools/LlamaTreeExp/diag 下两个语料
    python export_typing.py --mode full         # 导出「ctx + 上屏文本」（默认，贴近真实输入）
    python export_typing.py --mode segment      # 只导出上屏片段（无上下文，不推荐）

两种语料的字段不同，这里都处理：
    segments.jsonl  {"ctx": 光标前文, "segment": 本次提交, "kind": "commit"|"backspace", ...}
    corpus.jsonl    {"ctx": 光标前文, "typed": 上屏文本, "kind": "accept", ...}

为什么默认导「ctx + segment」：输入法的任务是「给定光标前文，预测下一个字」。
只导片段（如只有「的」）会丢掉任务结构，蒸馏出来的学生学不到条件分布。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "tools", "LlamaTreeExp"))
sys.path.insert(0, TREE)

import corpus as corpus_mod  # noqa: E402


def collect(path, key_file):
    """返回 [(ctx, text)]，跳过 backspace/无文本记录。"""
    if not os.path.exists(path):
        print("  跳过（不存在）：%s" % path)
        return []
    writer = corpus_mod.CorpusWriter(path, key_file=key_file)
    recs = writer.read_all()
    rows = []
    for r in recs:
        if str(r.get("kind", "commit")).startswith("backspace"):
            continue
        txt = r.get("segment") or r.get("typed") or r.get("text") or ""
        if not txt:
            continue
        rows.append((r.get("ctx") or "", txt))
    print("  %s：解密 %d 条，可用 %d 条" % (os.path.basename(path), len(recs), len(rows)))
    if rows:
        print("    样例：ctx=%r text=%r" % rows[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default=os.path.join(TREE, "diag"))
    ap.add_argument("--key-file", default=None, help="默认 <diag>/corpus.key")
    ap.add_argument("--out", default=os.path.join(HERE, "src", "typing.txt"))
    ap.add_argument("--mode", default="full", choices=["full", "segment"])
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    key_file = args.key_file or os.path.join(args.diag, "corpus.key")
    print("密钥文件：%s（存在=%s）" % (key_file, os.path.exists(key_file)))

    rows = []
    print("解密中：")
    rows += collect(os.path.join(args.diag, "segments.jsonl"), key_file)
    rows += collect(os.path.join(args.diag, "corpus.jsonl"), key_file)

    # 拼接成行：full 模式带上下文，segment 模式只有上屏文本
    lines = []
    for ctx, txt in rows:
        lines.append((ctx + txt) if args.mode == "full" else txt)

    # 去重（在线 accept 与离线 commit 会重叠）
    seen, uniq = set(), []
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    if args.max:
        uniq = uniq[:args.max]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(uniq))
    total = sum(len(s) for s in uniq)
    print("导出：%s（mode=%s）" % (args.out, args.mode))
    print("  去重后 %d 行 / %d 字；平均每行 %.1f 字"
          % (len(uniq), total, total / max(len(uniq), 1)))


if __name__ == "__main__":
    main()
