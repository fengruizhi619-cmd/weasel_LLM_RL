@echo off
chcp 65001 >nul
setlocal
set HERE=%~dp0
set DIAG=%HERE%diag
set READER=%HERE%..\WeaselExpContextV0\build\WeaselExpContextV0.exe
if not exist "%DIAG%" mkdir "%DIAG%"
echo [v0.2] cli_emojiless_exp_v0.2 - context + tree + RL
echo [v0.2] Switch to Notepad and type Chinese with Weasel.
echo.
start /B "" "%READER%" -n 100 -log "%DIAG%\exp-run.log" -diag "%DIAG%\exp-diag.log"
python "%HERE%tree_watcher_v02.py" --log-file "%DIAG%\exp-run.log" -n 5 -d 5 --top-n 5 --rl-lr 0.0001
taskkill /IM WeaselExpContextV0.exe /F >nul 2>&1
pause
