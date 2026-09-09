@echo off
chcp 65001 >nul
echo [online] stopping engine on 8081...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8081 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
echo [online] stopping context reader...
taskkill /IM WeaselExpContextV0.exe /F >nul 2>&1
echo [online] done. WeaselServer left running (normal input keeps working).
pause
