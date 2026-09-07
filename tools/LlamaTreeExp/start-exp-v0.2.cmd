@echo off

chcp 65001 >nul

setlocal

set HERE=%~dp0

set DIAG=%HERE%diag

set READER=%HERE%..\WeaselExpContextV0\build\WeaselExpContextV0.exe

echo [v0.3] version guard: checking WeaselRoot / server / user data
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%ensure-weasel-v0.3.ps1"
if errorlevel 1 (
  echo [v0.3] version guard failed, stop.
  pause
  exit /b 1
)

echo [v0.3] Switch to Notepad and type Chinese with Weasel.

echo.

start /B "" "%READER%" -n 100 -log "%DIAG%\exp-run.log" -diag "%DIAG%\exp-diag.log"

set PYTHONIOENCODING=utf-8

python -X utf8 "%HERE%unified_watcher.py" --log-file "%DIAG%\exp-run.log" -n 5 -d 5 --top-n 5 --rl-lr 0.0001 --ckpt-dir "%DIAG%\checkpoints"

taskkill /IM WeaselExpContextV0.exe /F >nul 2>&1

pause