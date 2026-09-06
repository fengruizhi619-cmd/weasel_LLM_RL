@echo off
chcp 65001 >nul
setlocal
set HERE=%~dp0
set EXE=%HERE%WeaselExpContextV0.exe
if not exist "%EXE%" (
  echo [exp-v0] exe not found: %EXE%
  exit /b 1
)
echo [exp-v0] cli_emojiless_exp_v0 - context reader (Ctrl+C to stop)
"%EXE%" -n 100
