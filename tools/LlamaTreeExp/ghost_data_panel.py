#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 数据面板：累计条数 + 离线训练入口（样本一次用完即丢）。

- 显示当前待训练条数、已训练条数、模式与进程状态
- 「训练并丢弃」把 segments.jsonl 交给 offline_train.py，训练成功后直接删掉
  这份数据（不重复使用样本），记录器会自动开始下一批
"""
import io
import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
SEGMENTS = os.path.join(HERE, "diag", "segments.jsonl")
SEEN = os.path.join(HERE, "diag", "segments.seen")
STATS = os.path.join(HERE, "diag", "train_stats.jsonl")
MODE_FILE = os.path.join(os.environ.get("APPDATA", ""), "Rime", "ghost_mode.txt")
TRAIN = os.path.join(HERE, "offline_train.py")
PYTHON = r"E:\python\python.exe"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with io.open(path, "rb") as f:
        return sum(1 for _ in f)


def read_mode():
    try:
        with io.open(MODE_FILE, encoding="utf-8-sig") as f:
            return f.read().strip() or "online"
    except Exception:
        return "online"


def process_alive(pattern):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
         "Where-Object { $_.CommandLine -like '*" + pattern + "*' } | "
         "ForEach-Object { $_.ProcessId }"],
        capture_output=True, text=True)
    return bool(out.stdout.strip())


def trained_count():
    n = 0
    if os.path.exists(STATS):
        with io.open(STATS, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    n += 1
    return n


class Panel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLM 数据面板 · cli_emojiless_RL")
        self.geometry("420x320")
        self.resizable(False, False)
        self.proc = None
        self.after_id = None

        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except Exception:
            pass

        self.mode_var = tk.StringVar()
        self.pending_var = tk.StringVar()
        self.trained_var = tk.StringVar()
        self.status_var = tk.StringVar()

        pad = {"padx": 12, "pady": 6, "sticky": "w"}
        ttk.Label(self, text="模式", width=10).grid(row=0, column=0, **pad)
        ttk.Label(self, textvariable=self.mode_var).grid(row=0, column=1, **pad)
        ttk.Label(self, text="待训练", width=10).grid(row=1, column=0, **pad)
        ttk.Label(self, textvariable=self.pending_var).grid(row=1, column=1, **pad)
        ttk.Label(self, text="已训练", width=10).grid(row=2, column=0, **pad)
        ttk.Label(self, textvariable=self.trained_var).grid(row=2, column=1, **pad)
        ttk.Label(self, text="进程", width=10).grid(row=3, column=0, **pad)
        ttk.Label(self, textvariable=self.status_var).grid(row=3, column=1, **pad)

        buttons = ttk.Frame(self)
        buttons.grid(row=4, column=0, columnspan=2, pady=10)
        self.train_btn = ttk.Button(buttons, text="训练并丢弃", command=self.train)
        self.train_btn.pack(side="left", padx=6)
        ttk.Button(buttons, text="刷新", command=self.refresh).pack(side="left", padx=6)
        ttk.Button(buttons, text="打开数据目录", command=self.open_dir).pack(side="left", padx=6)

        self.log_text = tk.Text(self, height=8, width=52, state="disabled",
                                background="#f7f7f7", relief="flat")
        self.log_text.grid(row=5, column=0, columnspan=2, padx=12, pady=(0, 10))

        self.refresh()
        self.log("样本一次用完即丢：训练成功后这份数据会被删除。")

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", time.strftime("[%H:%M:%S] ") + message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def open_dir(self):
        os.startfile(os.path.join(HERE, "diag"))

    def refresh(self):
        self.mode_var.set(read_mode())
        self.pending_var.set("%d 条" % count_lines(SEGMENTS))
        self.trained_var.set("%d 批" % trained_count())
        engine = "运行中" if process_alive("online_server.py") else "未运行"
        recorder = "运行中" if process_alive("offline_recorder.py") else "未运行"
        self.status_var.set("引擎 %s / 记录器 %s" % (engine, recorder))

    def train(self):
        if self.proc is not None:
            return
        if count_lines(SEGMENTS) == 0:
            messagebox.showinfo("LLM 数据面板", "当前没有待训练数据。")
            return
        if not messagebox.askyesno(
                "LLM 数据面板",
                "用当前这批数据训练一次，训练成功后数据会被删除（样本不重复使用）。继续？"):
            return
        self.train_btn.config(state="disabled", text="训练中…")
        self.log("开始离线训练…")
        self.proc = subprocess.Popen(
            [PYTHON, TRAIN], cwd=HERE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", creationflags=NO_WINDOW)
        self.after_id = self.after(500, self.poll)

    def poll(self):
        if self.proc is None:
            return
        line = self.proc.stdout.readline()
        while line:
            self.log(line.rstrip())
            line = self.proc.stdout.readline()
        if self.proc.poll() is None:
            self.after_id = self.after(500, self.poll)
            return
        code = self.proc.returncode
        self.proc = None
        self.train_btn.config(state="normal", text="训练并丢弃")
        if code == 0:
            removed = count_lines(SEGMENTS)
            try:
                if os.path.exists(SEGMENTS):
                    os.remove(SEGMENTS)
                if os.path.exists(SEEN):
                    os.remove(SEEN)
            except OSError as exc:
                self.log("删除数据失败: %r" % (exc,))
            with io.open(STATS, "a", encoding="utf-8") as f:
                f.write("%s trained=%d\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), removed))
            self.log("训练完成，已丢弃 %d 条样本。" % removed)
        else:
            self.log("训练失败（exit=%d），数据保留。" % code)
        self.refresh()


if __name__ == "__main__":
    Panel().mainloop()
