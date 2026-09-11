#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 数据面板：累计条数 + 离线训练入口（样本一次用完即丢）。

一条数据 = 一次提交断点。例如「你吃饭了吗」被输入法分四次上屏，
就记成 你 / 吃饭 / 了 / 吗 四条；回退记一条 backspace。

训练在独立进程里跑（CREATE_NO_WINDOW，无控制台窗口），输出由后台
线程读进队列，UI 线程用 after() 轮询，所以训练期间面板不会卡。
进度条按 offline_train.py 的 [progress] i/total 行更新。
"""
import io
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
SEGMENTS = os.path.join(HERE, "diag", "segments.jsonl")
SEEN = os.path.join(HERE, "diag", "segments.seen")
STATS = os.path.join(HERE, "diag", "train_stats.jsonl")
DISCARD_PREF = os.path.join(HERE, "diag", "train_discard.txt")
MODE_FILE = os.path.join(os.environ.get("APPDATA", ""), "Rime", "ghost_mode.txt")
TRAIN = os.path.join(HERE, "offline_train.py")
PYTHON = os.environ.get("WEASEL_LLM_PYTHON", "").strip() or "python.exe"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with io.open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def seen_count():
    try:
        with io.open(SEEN, encoding="utf-8") as f:
            return int(f.read().strip() or 0)
    except Exception:
        return 0


def pending_count():
    return max(0, count_lines(SEGMENTS) - seen_count())


def read_discard_pref():
    """True = drop the used records after a run (the original behaviour)."""
    try:
        with io.open(DISCARD_PREF, encoding="utf-8") as f:
            return f.read().strip() != "0"
    except Exception:
        return True


def write_discard_pref(value):
    try:
        with io.open(DISCARD_PREF, "w", encoding="utf-8") as f:
            f.write("1" if value else "0")
    except OSError:
        pass


def train_button_label(discard):
    return "训练并丢弃" if discard else "训练并保留"


def read_mode():
    try:
        with io.open(MODE_FILE, encoding="utf-8-sig") as f:
            return f.read().strip() or "online"
    except Exception:
        return "online"


def process_alive(pattern):
    script = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like "
              "\"*python*\" -and $_.CommandLine -like \"*" + pattern +
              "*\" } | ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         capture_output=True, text=True,
                         creationflags=NO_WINDOW)
    return bool(out.stdout.strip())


def trained_batches():
    if not os.path.exists(STATS):
        return 0
    with io.open(STATS, encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def trim_consumed(path, consumed):
    """Drop the first `consumed` records, keep anything written meanwhile."""
    if not os.path.exists(path):
        return 0
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    keep = [ln for ln in lines[consumed:] if ln.strip()]
    if keep:
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            f.writelines(keep)
    else:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.remove(SEEN)
    except OSError:
        pass
    return len(keep)


class Panel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLM 数据面板 · cli_emojiless_RL")
        self.geometry("480x470")
        self.resizable(False, False)
        self.proc = None
        self.queue = queue.Queue()
        self.trained_lines = 0

        try:
            ttk.Style(self).theme_use("vista")
        except Exception:
            pass

        self.mode_var = tk.StringVar()
        self.pending_var = tk.StringVar()
        self.trained_var = tk.StringVar()
        self.status_var = tk.StringVar(value="检测中…")
        self.progress_var = tk.StringVar(value="空闲")

        pad = {"padx": 12, "pady": 5, "sticky": "w"}
        ttk.Label(self, text="模式", width=8).grid(row=0, column=0, **pad)
        ttk.Label(self, textvariable=self.mode_var).grid(row=0, column=1, **pad)
        ttk.Label(self, text="待训练", width=8).grid(row=1, column=0, **pad)
        ttk.Label(self, textvariable=self.pending_var).grid(row=1, column=1, **pad)
        ttk.Label(self, text="已训练", width=8).grid(row=2, column=0, **pad)
        ttk.Label(self, textvariable=self.trained_var).grid(row=2, column=1, **pad)
        ttk.Label(self, text="进程", width=8).grid(row=3, column=0, **pad)
        ttk.Label(self, textvariable=self.status_var).grid(row=3, column=1, **pad)

        self.discard_var = tk.BooleanVar(value=read_discard_pref())
        opts = ttk.Frame(self)
        opts.grid(row=4, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 0))
        ttk.Checkbutton(opts, text="训练后丢弃已用数据", variable=self.discard_var,
                        command=self._on_discard_toggle).pack(side="left")
        ttk.Label(opts, text="（不勾选：保留在磁盘，但标记为不再参与训练）").pack(
            side="left", padx=6)

        buttons = ttk.Frame(self)
        buttons.grid(row=5, column=0, columnspan=2, pady=8)
        self.train_btn = ttk.Button(
            buttons, text=train_button_label(self.discard_var.get()),
            command=self.train)
        self.train_btn.pack(side="left", padx=6)
        ttk.Button(buttons, text="刷新", command=self.refresh).pack(side="left", padx=6)
        ttk.Button(buttons, text="打开数据目录", command=self.open_dir).pack(side="left", padx=6)

        ttk.Label(self, textvariable=self.progress_var).grid(
            row=6, column=0, columnspan=2, padx=12, sticky="w")
        self.progress = ttk.Progressbar(self, mode="determinate", length=450)
        self.progress.grid(row=7, column=0, columnspan=2, padx=12, pady=(0, 6))

        self.log_text = tk.Text(self, height=11, width=60, state="disabled",
                                background="#f7f7f7", relief="flat")
        self.log_text.grid(row=8, column=0, columnspan=2, padx=12, pady=(0, 10))

        self.refresh()
        self.log("一条数据 = 一次提交断点：你 / 吃饭 / 了 / 吗 = 4 条；回退 = 1 条。")
        self.log("训练在独立无窗口进程里跑，面板不卡；训练完按上面的勾选处置数据。")
        self.after(200, self.drain)

    def _on_discard_toggle(self):
        write_discard_pref(self.discard_var.get())
        self.train_btn.config(text=train_button_label(self.discard_var.get()))

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", time.strftime("[%H:%M:%S] ") + message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def open_dir(self):
        os.startfile(os.path.join(HERE, "diag"))

    def refresh(self):
        self.mode_var.set(read_mode())
        self.pending_var.set("%d 条" % pending_count())
        self.trained_var.set("%d 批" % trained_batches())
        self.detect_processes()

    def detect_processes(self):
        self.status_var.set("检测中…")

        def worker():
            engine = "运行中" if process_alive("online_server.py") else "未运行"
            recorder = "运行中" if process_alive("offline_recorder.py") else "未运行"
            self.queue.put(("status", "引擎 %s / 记录器 %s" % (engine, recorder)))

        threading.Thread(target=worker, daemon=True).start()

    def drain(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if item is None:
                    self.finish()
                    continue
                kind, payload = item
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "line":
                    self.handle_line(payload)
        except queue.Empty:
            pass
        self.after(200, self.drain)

    def train(self):
        if self.proc is not None:
            return
        total = count_lines(SEGMENTS)
        pending = pending_count()
        if pending == 0:
            messagebox.showinfo("LLM 数据面板", "当前没有待训练数据。")
            return
        if not messagebox.askyesno(
                "LLM 数据面板",
                "用这 %d 条数据训练一次？训练成功后这批样本会被丢弃。\n"
                "训练在后台无窗口进程里跑，面板可以继续使用。" % pending):
            return
        self.trained_lines = total
        self.train_btn.config(state="disabled", text="训练中…")
        self.progress["maximum"] = max(1, pending)
        self.progress["value"] = 0
        self.progress_var.set("准备中…（加载模型约 30 秒）")
        self.log("开始离线训练：%d 条（无窗口）" % pending)
        # [TRAIN-032] Train the way the live path does, and score every
        # recorded character exactly once. --include-ctx did the opposite: it
        # re-scored the context that consecutive records share, so 375 records
        # became 3519 steps covering only 168 distinct characters (~21x each)
        # and wore the head's discrimination away.
        train_cmd = [PYTHON, TRAIN, "--mode", "stream"]
        self.log("训练命令：" + " ".join(train_cmd[1:]))
        # [CTX-005] Popen can fail (python.exe not on PATH, model folder moved,
        # no permission). It used to raise straight out of the Tk callback, so
        # the button stayed on "训练中…" for ever and nothing said why.
        try:
            self.proc = subprocess.Popen(
                train_cmd, cwd=HERE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", creationflags=NO_WINDOW)
        except Exception as exc:
            self.proc = None
            self.train_btn.config(
                state="normal", text=train_button_label(self.discard_var.get()))
            self.progress_var.set("启动失败")
            self.log("[错误] 训练进程启动失败：%r" % (exc,))
            messagebox.showerror(
                "LLM 数据面板",
                "训练进程启动失败：\n%r\n\n检查 python 是否可用（WEASEL_LLM_PYTHON）"
                % (exc,))
            return
        threading.Thread(target=self._reader, args=(self.proc,),
                         daemon=True).start()

    def _reader(self, proc):
        try:
            for line in proc.stdout:
                self.queue.put(("line", line.rstrip()))
        finally:
            self.queue.put(None)

    def handle_line(self, line):
        if line.startswith("[progress] "):
            parts = line.split()
            try:
                current, total = parts[1].split("/")
                self.progress["maximum"] = max(1, int(total))
                self.progress["value"] = int(current)
                extra = " ".join(parts[2:])
                self.progress_var.set("训练中 %s / %s 条  %s"
                                      % (current, total, extra))
            except Exception:
                pass
        elif line.startswith("[offline] done") or line.startswith("[offline] "):
            self.progress_var.set(line)
        self.log(line)

    def finish(self):
        proc = self.proc
        code = -1
        if proc is not None:
            try:
                code = proc.wait(timeout=15)
            except Exception:
                code = proc.returncode if proc.returncode is not None else -1
        self.proc = None
        self.train_btn.config(state="normal",
                              text=train_button_label(self.discard_var.get()))
        if code == 0:
            if self.discard_var.get():
                kept = trim_consumed(SEGMENTS, self.trained_lines)
                note = "训练 %d 条并丢弃，期间新增保留 %d 条" % (self.trained_lines, kept)
            else:
                # [TRAIN-030] Keep the records on disk but mark them consumed, so
                # they stay available for inspection without ever being trained
                # on a second time.
                with io.open(SEEN, "w", encoding="utf-8") as f:
                    f.write(str(self.trained_lines))
                kept = max(0, count_lines(SEGMENTS) - self.trained_lines)
                note = ("训练 %d 条并保留在磁盘（已标记，不再参与训练），"
                        "期间新增待训练 %d 条" % (self.trained_lines, kept))
            with io.open(STATS, "a", encoding="utf-8") as f:
                f.write("%s trained=%d kept=%d\n"
                        % (time.strftime("%Y-%m-%d %H:%M:%S"),
                           self.trained_lines, kept))
            self.progress_var.set("完成：" + note)
            self.log("训练完成：" + note + "。")
        else:
            self.progress_var.set("训练失败（exit=%d），数据保留" % code)
            self.log("训练失败（exit=%d），数据保留。" % code)
        self.refresh()


if __name__ == "__main__":
    Panel().mainloop()