@echo off
chcp 65001 >nul
setlocal
set HERE=%~dp0
set DIAG=%HERE%diag
set READER=%HERE%..\WeaselExpContextV0\build\WeaselExpContextV0.exe
set WEASEL_ROOT=%ProgramFiles%\Rime\weasel-0.17.4-emoji-off
if not exist "%DIAG%" mkdir "%DIAG%"

echo [online] 1/3 WeaselServer
tasklist /FI "IMAGENAME eq WeaselServer.exe" 2>nul | find /I "WeaselServer.exe" >nul
if errorlevel 1 start "" "%WEASEL_ROOT%\WeaselServer.exe"

echo [online] 2/3 context reader (WeaselExpContextV0)
tasklist /FI "IMAGENAME eq WeaselExpContextV0.exe" 2>nul | find /I "WeaselExpContextV0.exe" >nul
if errorlevel 1 start /B "" "%READER%" -n 100 -log "%DIAG%\exp-run-v02.log" -diag "%DIAG%\exp-diag-v02.log"

echo [online] 3/3 unified engine on 127.0.0.1:8081 (Ctrl+C to stop)
python "%HERE%online_server.py" --log-file "%DIAG%\exp-run-v02.log" --ckpt-dir "%DIAG%\checkpoints_online" --corpus "%DIAG%\corpus.jsonl" --dtype float16 -n 20 -d 2

taskkill /IM WeaselExpContextV0.exe /F >nul 2>&1
echo [online] stopped.
pause
