@echo off
chcp 65001 >nul
setlocal
set HERE=%~dp0
set DIAG=%HERE%diag
if not exist "%DIAG%" mkdir "%DIAG%"

echo [v0.2] cli_emojiless_exp_v0.2
echo [v0.2] context reader + candidate tree
echo.

rem Start C# context reader in background (writes to log file)
start /B "" "%HERE%..\WeaselExpContextV0\build\WeaselExpContextV0.exe" -n 100 -log "%DIAG%\exp-run.log" -diag "%DIAG%\exp-diag.log"

rem Start Python tree watcher (monitors log, builds trees)
python "%HERE%tree_watcher.py" --log-file "%DIAG%\exp-run.log" -n 5 -d 5 --top-n 5

rem When watcher exits, stop the C# reader too
taskkill /IM WeaselExpContextV0.exe /F >nul 2>&1
pause
