$ErrorActionPreference = 'Stop'
$base = Split-Path -Parent $PSCommandPath
$log = Join-Path $base 'install_ghost.log'
$src = Join-Path $base 'weaselx64.dll'
$dir = 'C:\Program Files\Rime\weasel-0.17.4-emoji-off'
$dst = Join-Path $dir 'weaselx64.dll'
$backup = Join-Path $dir 'weaselx64.dll.v0.3.bak'
try {
  Get-Process -Name WeaselServer -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep -Milliseconds 800
  if (Test-Path -LiteralPath $dst) {
    Copy-Item -LiteralPath $dst -Destination $backup -Force
  }
  Copy-Item -LiteralPath $src -Destination $dst -Force
  
  $runKey = 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run'
  if (-not (Test-Path $runKey)) { New-Item -Path $runKey -Force | Out-Null }
  Set-ItemProperty -Path $runKey -Name 'WeaselServer' -Value (Join-Path $dir 'WeaselServer.exe') -Force
  Set-Content -LiteralPath $log -Value 'INSTALL_OK'
} catch {
  Set-Content -LiteralPath $log -Value ('INSTALL_FAIL: ' + $_.Exception.Message)
  exit 1
}