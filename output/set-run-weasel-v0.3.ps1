$ErrorActionPreference = 'Stop'
$runKey = 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run'
$dir = 'C:\Program Files\Rime\weasel-0.17.4-emoji-off'
if (-not (Test-Path $runKey)) { New-Item -Path $runKey -Force | Out-Null }
Set-ItemProperty -Path $runKey -Name 'WeaselServer' -Value (Join-Path $dir 'WeaselServer.exe') -Force
Write-Host 'Run key set to weasel-0.17.4-emoji-off'