@echo off
chcp 936 >nul
set "GHOST_HOME="
if exist "%~dp0ghost_home.txt" set /p GHOST_HOME=<"%~dp0ghost_home.txt"
if not defined GHOST_HOME set "GHOST_HOME=%WEASEL_LLM_HOME%"
if not defined GHOST_HOME set "GHOST_HOME=%~dp0..\.."
cd /d "%GHOST_HOME%\tools\LlamaTreeExp"
set "PYTHONW=pythonw.exe"
if defined WEASEL_LLM_PYTHONW set "PYTHONW=%WEASEL_LLM_PYTHONW%"
set "MODEFILE=%APPDATA%\Rime\ghost_mode.txt"
set "MODE=online"
if exist "%MODEFILE%" for /f "usebackq tokens=*" %%m in ("%MODEFILE%") do set "MODE=%%m"
if /I "%MODE%"=="offline" (
  echo [ghost] offline mode: recording context + segments only
  "%PYTHONW%" offline_recorder.py --log-file "diag\exp-run-v02.log" --out "diag\segments.jsonl" >> "diag\offline-recorder.log" 2>&1
) else (
  echo [ghost] online mode: prediction service on 127.0.0.1:8081
  "%PYTHONW%" online_server.py --log-file "diag\exp-run-v02.log" --ckpt-dir "diag\checkpoints_online" --corpus "diag\corpus.jsonl" --dtype float16 -n 20 -d 10 >> "diag\online-server.log" 2>> "diag\online-server.err.log"
)
