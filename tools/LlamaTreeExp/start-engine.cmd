@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" /B "E:\python\pythonw.exe" online_server.py --log-file "diag\exp-run-v02.log" --ckpt-dir "diag\checkpoints_online" --corpus "diag\corpus.jsonl" --dtype float16 -n 20 -d 10 >> "diag\online-server.log" 2>> "diag\online-server.err.log"
