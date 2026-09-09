@echo off
set "GHOST_HOME="
if exist "%~dp0ghost_home.txt" set /p GHOST_HOME=<"%~dp0ghost_home.txt"
if not defined GHOST_HOME set "GHOST_HOME=%WEASEL_LLM_HOME%"
if not defined GHOST_HOME set "GHOST_HOME=%~dp0..\.."
set "PYTHONW=pythonw.exe"
if defined WEASEL_LLM_PYTHONW set "PYTHONW=%WEASEL_LLM_PYTHONW%"
start "LLM Data Panel" "%PYTHONW%" "%GHOST_HOME%\tools\LlamaTreeExp\ghost_data_panel.py"
