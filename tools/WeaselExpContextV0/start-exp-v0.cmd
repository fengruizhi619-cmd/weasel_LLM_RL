@echo off
chcp 65001 >nul
setlocal
set HERE=%~dp0
if not exist "%HERE%diag" mkdir "%HERE%diag"
echo [exp-v0] logging to %HERE%diag\exp-run.log (Ctrl+C to stop)
"%HERE%WeaselExpContextV0.exe" -n 100 -log "%HERE%diag\exp-run.log" -diag "%HERE%diag\exp-diag.log"
echo [exp-v0] logs: %HERE%diag\exp-run.log / exp-diag.log
