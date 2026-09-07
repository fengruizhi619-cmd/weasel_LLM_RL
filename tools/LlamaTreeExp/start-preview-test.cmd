@echo off
chcp 65001 >nul
setlocal
set HERE=%~dp0
set DIAG=%HERE%diag
set TSFLOG=%APPDATA%\Rime\llm_tsf_context.log
if exist "%TSFLOG%" del "%TSFLOG%"
if not exist "%DIAG%" mkdir "%DIAG%"
echo [preview-test] version guard
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%ensure-weasel-v0.3.ps1"
if errorlevel 1 pause & exit /b 1
echo [preview-test] skip ckpt, base lm_head, watch TSF context
set PYTHONIOENCODING=utf-8
python -X utf8 "%HERE%unified_watcher.py" --log-file "%TSFLOG%" -n 5 -d 5 --top-n 5 --rl-lr 0.0001 --ckpt-dir "%DIAG%\checkpoints" --skip-ckpt
pause