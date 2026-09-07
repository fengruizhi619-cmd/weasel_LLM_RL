$ErrorActionPreference = 'Stop'
$runtimeDir = 'E:\codex_data\研究\weasel-baseline\output\v0.3-tsf-runtime12'
$sourceDll = 'E:\codex_data\研究\weasel-baseline\output\weaselx64.dll'
$clsid = '{A3F4CDED-B1E9-41EE-9CA6-7B4D0DE6CB0A}'
if (-not (Test-Path -LiteralPath $runtimeDir)) { New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null }
$dll = Join-Path $runtimeDir 'weaselx64.dll'
Copy-Item -LiteralPath $sourceDll -Destination $dll -Force
$key = 'HKLM:\SOFTWARE\Classes\CLSID\' + $clsid + '\InprocServer32'
if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }
New-ItemProperty -Path $key -Name '(default)' -Value $dll -PropertyType String -Force | Out-Null
Set-ItemProperty -Path $key -Name 'ThreadingModel' -Value 'Apartment' -Force
Write-Host ('TSF CLSID -> ' + $dll)