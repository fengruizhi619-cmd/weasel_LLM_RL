#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解码器（lm_head）的库与切换。

    diag/heads/<name>.pt      头文件本身（与 checkpoints_online 里的同格式）
    diag/heads/<name>.json    元数据：note / updates / ts / 导入时间

切换 = 把选中的头写进 checkpoints_online/lm_head_t0.pt。两个细节：

  1. 切换前先把「当前线上那份」收进库（按 updates+timestamp 判重），所以随时能换回来；
  2. 写入时把 updates 提成「该目录所有槽的最大值 + 1」。在线服务每 15 秒会
     sync_from_disk() 一次，只有磁盘上的 updates 更大才会采纳——不提这一格，
     换过去也不会生效，非得重启服务。

命令行：
    python head_lib.py list
    python head_lib.py import <name> <path.pt> [note]
    python head_lib.py snapshot            # 把当前线上那份收进库
    python head_lib.py activate <name>
"""
import io
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.join(HERE, "diag")
LIB = os.path.join(DIAG, "heads")
ONLINE = os.path.join(DIAG, "checkpoints_online")
SLOTS = ["lm_head_t0.pt", "lm_head_t5m.pt", "lm_head_t25m.pt",
         "lm_head_t2h.pt", "lm_head_t12h.pt"]


def _load_meta(name):
    p = os.path.join(LIB, name + ".json")
    if os.path.exists(p):
        try:
            return json.load(io.open(p, encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_meta(name, meta):
    os.makedirs(LIB, exist_ok=True)
    with io.open(os.path.join(LIB, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _ckpt_info(path):
    """只读元数据，不加载整份权重。"""
    import torch
    try:
        m = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        return {"updates": m.get("updates"), "ts": m.get("timestamp"),
                "dtype": m.get("dtype"), "size": os.path.getsize(path)}
    except Exception as e:
        return {"updates": None, "ts": None, "dtype": None,
                "size": os.path.getsize(path), "error": repr(e)}


def list_heads():
    """库里所有的头 + 当前线上那份，按更新时间倒序。"""
    os.makedirs(LIB, exist_ok=True)
    out = []
    for f in sorted(os.listdir(LIB)):
        if not f.endswith(".pt"):
            continue
        name = f[:-3]
        info = _ckpt_info(os.path.join(LIB, f))
        info.update(_load_meta(name))
        info["name"] = name
        info["mtime"] = os.path.getmtime(os.path.join(LIB, f))
        out.append(info)
    out.sort(key=lambda d: d.get("mtime", 0), reverse=True)
    live = os.path.join(ONLINE, "lm_head_t0.pt")
    active = _ckpt_info(live) if os.path.exists(live) else None
    return out, active


def _dup_in_lib(updates, ts):
    for f in os.listdir(LIB) if os.path.isdir(LIB) else []:
        if not f.endswith(".pt"):
            continue
        m = _load_meta(f[:-3])
        if m.get("updates") == updates and str(m.get("ts")) == str(ts):
            return f[:-3]
    return None


def snapshot(name=None, note="切换前自动备份"):
    """把当前线上那份收进库；已存在（updates+ts 相同）则跳过。"""
    live = os.path.join(ONLINE, "lm_head_t0.pt")
    if not os.path.exists(live):
        return None, "线上没有 lm_head_t0.pt"
    info = _ckpt_info(live)
    dup = _dup_in_lib(info.get("updates"), info.get("ts"))
    if dup:
        return dup, "当前这份已在库里：%s" % dup
    name = name or ("auto_%s" % time.strftime("%m%d-%H%M%S"))
    os.makedirs(LIB, exist_ok=True)
    shutil.copy2(live, os.path.join(LIB, name + ".pt"))
    _save_meta(name, {"note": note, "updates": info.get("updates"),
                      "ts": info.get("ts"), "dtype": info.get("dtype"),
                      "imported": time.strftime("%Y-%m-%d %H:%M:%S")})
    return name, "已把当前线上头收进库：%s（updates=%s）" % (name, info.get("updates"))


def import_head(name, path, note=""):
    if not os.path.exists(path):
        return "找不到：%s" % path
    os.makedirs(LIB, exist_ok=True)
    shutil.copy2(path, os.path.join(LIB, name + ".pt"))
    info = _ckpt_info(path)
    _save_meta(name, {"note": note, "updates": info.get("updates"),
                      "ts": info.get("ts"), "dtype": info.get("dtype"),
                      "imported": time.strftime("%Y-%m-%d %H:%M:%S")})
    return "已导入 %s（updates=%s）" % (name, info.get("updates"))


def max_updates():
    best = 0
    for s in SLOTS:
        p = os.path.join(ONLINE, s)
        if os.path.exists(p):
            u = _ckpt_info(p).get("updates") or 0
            best = max(best, int(u))
    return best


def activate(name):
    src = os.path.join(LIB, name + ".pt")
    if not os.path.exists(src):
        return "库里没有 %s" % name
    os.makedirs(ONLINE, exist_ok=True)
    snap, msg = snapshot()
    import torch
    new_updates = max_updates() + 1
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    ckpt["updates"] = new_updates
    ckpt["timestamp"] = time.strftime("%Y%m%d_%H%M%S")
    tmp = os.path.join(ONLINE, "lm_head_t0.pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, os.path.join(ONLINE, "lm_head_t0.pt"))
    return ("已切换为 %s（updates=%d，比原来最大槽高 1，在线服务 15 秒内会自动采纳）；%s"
            % (name, new_updates, msg))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "list":
        heads, active = list_heads()
        print("库里：")
        for h in heads:
            print("  %-24s updates=%-6s %-16s %6.1f MB  %s"
                  % (h["name"], h.get("updates"), h.get("ts"),
                     h.get("size", 0) / 1024**2, h.get("note", "")))
        print("当前线上：")
        if active:
            print("  lm_head_t0.pt            updates=%-6s %-16s %6.1f MB"
                  % (active.get("updates"), active.get("ts"), active.get("size", 0) / 1024**2))
        else:
            print("  （没有）")
    elif cmd == "import":
        print(import_head(sys.argv[2], sys.argv[3],
                          " ".join(sys.argv[4:]) if len(sys.argv) > 4 else ""))
    elif cmd == "snapshot":
        print(snapshot()[1])
    elif cmd == "activate":
        print(activate(sys.argv[2]))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
