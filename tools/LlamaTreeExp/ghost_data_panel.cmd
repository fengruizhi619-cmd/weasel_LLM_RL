@echo off
rem ghost_data_panel.cmd - open the data panel (installed copy carries the repo path)
rem
rem WHY THE PATH IS INLINE: this file used to read the repo root from ghost_home.txt via
rem `set /p`. cmd reads that with the ANSI code page, so the moment the file is written as
rem UTF-8 the non-ASCII characters in the path turn into mojibake and pythonw silently
rem launches nothing. The installer now substitutes @@GHOST_HOME@@ below with the repo
rem root, so the path lives in this file as plain ASCII and there is no encoding to get wrong.
rem It is also an ASCII literal on purpose: keep this file ASCII-only (no BOM).
rem
rem Do NOT commit a substituted copy: keep @@GHOST_HOME@@ in the template.

set "GHOST_HOME=@@GHOST_HOME@@"
if not defined GHOST_HOME set "GHOST_HOME=%WEASEL_LLM_HOME%"
if not defined GHOST_HOME set "GHOST_HOME=%~dp0..\.."
set "PYTHONW=pythonw.exe"
if defined WEASEL_LLM_PYTHONW set "PYTHONW=%WEASEL_LLM_PYTHONW%"
start "LLM Data Panel" "%PYTHONW%" "%GHOST_HOME%\tools\LlamaTreeExp\ghost_data_panel.py"
