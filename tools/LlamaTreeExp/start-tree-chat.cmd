@echo off
chcp 65001 >nul
setlocal
set HERE=%~dp0
set /P PROMPT_TEXT="Enter prompt text: "
python "%HERE%tree_exp.py" -n 5 -d 5 --text "%PROMPT_TEXT%" --model "%HERE%..\..\models\Qwen3-0.6B-Chat-Q8_0.gguf"
pause
