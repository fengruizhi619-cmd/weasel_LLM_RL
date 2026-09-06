# build.ps1 - compile WeaselExpContextV0 (cli_emojiless_exp_v0) with .NET Framework csc
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $here 'WeaselExpContextV0.cs'
$outDir = Join-Path $here 'build'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$out = Join-Path $outDir 'WeaselExpContextV0.exe'
$csc = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path $csc)) { $csc = 'C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe' }
if (-not (Test-Path $csc)) { throw 'csc.exe not found' }
function Get-GacDll($name) {
  $root = Join-Path $env:WINDIR ("Microsoft.NET\assembly\GAC_MSIL\" + $name)
  $d = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $d) { throw ("GAC dll not found: " + $name) }
  $dll = Get-ChildItem $d.FullName -Filter '*.dll' | Select-Object -First 1
  return $dll.FullName
}
$refUiaClient = Get-GacDll 'UIAutomationClient'
$refUiaTypes = Get-GacDll 'UIAutomationTypes'
$refWinBase = Get-GacDll 'WindowsBase'
& $csc /nologo /target:exe /out:$out /r:$refUiaClient /r:$refUiaTypes /r:$refWinBase $src
if ($LASTEXITCODE -ne 0) { throw ('compile failed: ' + $LASTEXITCODE) }
Write-Output ("built " + $out + " (" + (Get-Item $out).Length + " bytes)")
