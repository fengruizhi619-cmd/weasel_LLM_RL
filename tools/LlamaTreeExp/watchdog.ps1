# watchdog.ps1 - keep the online chain alive
$ErrorActionPreference = 'Continue'
$base = 'E:\codex_data\研究\weasel-baseline'
$exp = Join-Path $base 'tools\LlamaTreeExp'
$diag = Join-Path $exp 'diag'
$log = Join-Path $diag 'watchdog.log'
New-Item -ItemType Directory -Force -Path $diag | Out-Null
function Note([string]$m) { Add-Content -LiteralPath $log -Value ((Get-Date -Format 'MM-dd HH:mm:ss') + ' ' + $m) -Encoding UTF8 }
if (-not (Get-Process WeaselServer -ErrorAction SilentlyContinue)) {
  $exe = 'C:\Program Files\Rime\weasel-0.17.4-emoji-off\WeaselServer.exe'
  if (Test-Path $exe) { Start-Process -FilePath $exe -WindowStyle Hidden; Note 'started WeaselServer' }
}
if (-not (Get-Process WeaselExpContextV0 -ErrorAction SilentlyContinue)) {
  $hook = Join-Path $base 'tools\WeaselExpContextV0\build\WeaselExpContextV0.exe'
  if (Test-Path $hook) {
    Start-Process -FilePath $hook -ArgumentList '-n','256','-log',(Join-Path $diag 'exp-run-v02.log'),'-diag',(Join-Path $diag 'exp-diag-v02.log') -WindowStyle Hidden
    Note 'started context hook'
  }
}
# the engine is now a child of WeaselServer; nothing to do here
$health = $null
try { $health = (Invoke-WebRequest -Uri 'http://127.0.0.1:8081/health' -UseBasicParsing -TimeoutSec 5).StatusCode } catch {}
if ($health -ne 200) { Note ('engine health=' + $health) }
# 3) background segment recorder: must run no matter which input method is used
$lock = Join-Path $diag 'recorder.lock'
$recAlive = $false
if (Test-Path $lock) {
  $rpid = Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($rpid) { $recAlive = [bool](Get-Process -Id ([int]$rpid) -ErrorAction SilentlyContinue) }
}
$recProc = Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Where-Object { $_.CommandLine -like '*offline_recorder.py*' }
if (-not $recAlive -and -not $recProc) {
  $rargs = @((Join-Path $exp 'offline_recorder.py'), '--log-file', (Join-Path $diag 'exp-run-v02.log'), '--out', (Join-Path $diag 'segments.jsonl'))
  Start-Process -FilePath 'E:\python\pythonw.exe' -ArgumentList $rargs -WorkingDirectory $exp -WindowStyle Hidden
  Note 'started background segment recorder'
}
