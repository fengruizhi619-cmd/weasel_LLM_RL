#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mode switch: online inference vs offline recording.

  python ghost_mode.py status
  python ghost_mode.py online
  python ghost_mode.py offline

The mode is written to %APPDATA%\\Rime\\ghost_mode.txt; WeaselServer reads it
through ghost_service.cmd when it starts, so the prediction service or the
segment recorder runs as a child of the input method either way.
"""
import argparse
import io
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODE_FILE = os.path.join(os.environ.get("APPDATA", ""), "Rime", "ghost_mode.txt")
WEASEL_SERVER = r"C:\Program Files\Rime\weasel-0.17.4-emoji-off\WeaselServer.exe"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def read_mode():
    try:
        with io.open(MODE_FILE, encoding="utf-8-sig") as f:
            value = f.read().strip().lower()
        if value:
            return value
    except Exception:
        pass
    return "online"


def write_mode(mode):
    os.makedirs(os.path.dirname(MODE_FILE), exist_ok=True)
    with io.open(MODE_FILE, "w", encoding="utf-8") as f:
        f.write(mode + "\n")


def ps(command):
    out = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                         capture_output=True, text=True)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def procs_by_cmd(marker):
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
           "Where-Object { $_.CommandLine -like '*" + marker + "*' } | "
           "ForEach-Object { $_.ProcessId }")
    return [int(x) for x in ps(cmd) if x.isdigit()]


def port_owner(port):
    out = ps("Get-NetTCPConnection -LocalPort " + str(port) +
             " -State Listen -ErrorAction SilentlyContinue | "
             "Select-Object -First 1 -ExpandProperty OwningProcess")
    return int(out[0]) if out and out[0].isdigit() else None


def kill(pid):
    if pid:
        ps("Stop-Process -Id " + str(pid) + " -Force -ErrorAction SilentlyContinue")


def engine_pids():
    pids = set(procs_by_cmd("online_server.py"))
    owner = port_owner(8081)
    if owner:
        pids.add(owner)
    return sorted(pids)


def job_children():
    """pythonw processes hosted by WeaselServer (their command line can be empty)."""
    out = ps("$ws = Get-Process WeaselServer -ErrorAction SilentlyContinue | "
             "Select-Object -First 1; if ($ws) { "
             "$cmds = Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | "
             "Where-Object { $_.ParentProcessId -eq $ws.Id }; "
             "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" | "
             "Where-Object { $cmds.ProcessId -contains $_.ParentProcessId } | "
             "ForEach-Object { $_.ProcessId } }")
    return [int(x) for x in out if x.isdigit()]


def recorder_pids():
    pids = set(procs_by_cmd("offline_recorder.py"))
    for pid in job_children():
        if pid not in engine_pids():
            pids.add(pid)
    return sorted(pids)


def weasel_pids():
    return ps("Get-Process WeaselServer -ErrorAction SilentlyContinue | "
              "ForEach-Object { $_.Id }")


def start_recorder():
    """The segment recorder is background collection: it must survive mode
    switches and run for every input method."""
    cmd = [r"E:\python\pythonw.exe", os.path.join(HERE, "offline_recorder.py"),
           "--log-file", os.path.join(HERE, "diag", "exp-run-v02.log"),
           "--out", os.path.join(HERE, "diag", "segments.jsonl")]
    subprocess.Popen(cmd, cwd=HERE, close_fds=True,
                     creationflags=NO_WINDOW,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def restart_weasel():
    for pid in weasel_pids():
        kill(int(pid))
    time.sleep(1.5)
    if os.path.exists(WEASEL_SERVER):
        subprocess.Popen([WEASEL_SERVER], close_fds=True, creationflags=NO_WINDOW)
        return True
    return False


def health():
    out = ps("try { (Invoke-WebRequest -Uri 'http://127.0.0.1:8081/health' "
             "-UseBasicParsing -TimeoutSec 3).StatusCode } catch { 'down' }")
    return out[0] if out else "down"


def status():
    print("mode      : %s" % read_mode())
    print("mode file : %s" % MODE_FILE)
    print("weasel    : %s" % (weasel_pids() or "not running"))
    print("engine    : %s (health=%s)" % (engine_pids() or "not running", health()))
    print("recorder  : %s" % (recorder_pids() or "not running"))
    segments = os.path.join(HERE, "diag", "segments.jsonl")
    if os.path.exists(segments):
        with io.open(segments, "rb") as f:
            print("segments  : %d lines (%d bytes)" %
                  (sum(1 for _ in f), os.path.getsize(segments)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["status", "online", "offline"])
    args = ap.parse_args()
    if args.mode == "status":
        status()
        return 0

    write_mode(args.mode)
    for pid in engine_pids():
        kill(pid)
    if not recorder_pids():
        start_recorder()
    time.sleep(0.5)
    restarted = restart_weasel()
    time.sleep(2.0)
    print("mode=%s  weasel_restarted=%s" % (args.mode, restarted))
    print("(WeaselServer hosts %s; the segment recorder runs in the "
          "background for every input method)" %
          ("offline_recorder.py" if args.mode == "offline" else "online_server.py"))
    print("waiting for it to come up...")
    for _ in range(45):
        time.sleep(2)
        if args.mode == "offline":
            if recorder_pids():
                break
        elif health() == "200" and recorder_pids():
            break
    status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
