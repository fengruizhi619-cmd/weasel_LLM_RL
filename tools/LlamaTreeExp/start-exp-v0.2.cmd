@echo off

chcp 65001 >nul

setlocal

set HERE=%~dp0

set DIAG=%HERE%diag

set READER=%HERE%..\WeaselExpContextV0\build\WeaselExpContextV0.exe

set WEASEL_ROOT=
for /f "tokens=2,*" %%A in ('reg query "HKLM\Software\Rime\Weasel" /v WeaselRoot 2^>nul ^| findstr /I "WeaselRoot"') do set "WEASEL_ROOT=%%~B"
if not defined WEASEL_ROOT set "WEASEL_ROOT=%ProgramFiles%\Rime\weasel-0.17.4-emoji-off"
set SERVER_EXE=%WEASEL_ROOT%\WeaselServer.exe

if not exist "%DIAG%" mkdir "%DIAG%"

echo [v0.2] cli_emojiless_exp_v0.2 - one-click start: Weasel + context + tree + RL

tasklist /FI "IMAGENAME eq WeaselServer.exe" 2>nul | find /I "WeaselServer.exe" >nul
if errorlevel 1 (
  echo [v0.2] starting WeaselServer from %SERVER_EXE%
  start "" "%SERVER_EXE%"
  timeout /t 1 /nobreak >nul
) else (
  echo [v0.2] WeaselServer already running
)

echo [v0.2] Switch to Notepad and type Chinese with Weasel.

echo.

start /B "" "%READER%" -n 100 -log "%DIAG%\exp-run.log" -diag "%DIAG%\exp-diag.log"

python "%HERE%unified_watcher.py" --log-file "%DIAG%\exp-run.log" -n 5 -d 5 --top-n 5 --rl-lr 0.0001 --ckpt-dir "%DIAG%\checkpoints"

taskkill /IM WeaselExpContextV0.exe /F >nul 2>&1

pause