@echo off
chcp 65001 >nul
cd /d "E:\codex_data\研究\weasel-baseline\tools\LlamaTreeExp"
"E:\python\pythonw.exe" online_server.py --log-file "diag\exp-run-v02.log" --ckpt-dir "diag\checkpoints_online" --corpus "diag\corpus.jsonl" --dtype float16 -n 20 -d 10 >> "diag\online-server.log" 2>> "diag\online-server.err.log"
