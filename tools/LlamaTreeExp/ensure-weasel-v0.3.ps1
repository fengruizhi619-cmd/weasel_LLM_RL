$ErrorActionPreference = 'Stop'
$expectedVersion = 'cli_emojiless_exp_v0.3'
$rimeKey = 'HKCU:\Software\Rime\Weasel'
$lmKey = 'HKLM:\Software\Rime\Weasel'

function Get-ProcessImageName([int]$pidValue) {
  Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class WeaselPathQuery {
  [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint access, bool inherit, uint processId);
  [DllImport("kernel32.dll")] public static extern bool QueryFullProcessImageName(IntPtr process, uint flags, StringBuilder name, ref uint size);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
}
"@ -ErrorAction SilentlyContinue
  $handle = [WeaselPathQuery]::OpenProcess(0x1000, $false, [uint32]$pidValue)
  if ($handle -eq [IntPtr]::Zero) { return '' }
  $builder = New-Object System.Text.StringBuilder 1024
  $length = [uint32]$builder.Capacity
  $ok = [WeaselPathQuery]::QueryFullProcessImageName($handle, 0, $builder, [ref]$length)
  [void][WeaselPathQuery]::CloseHandle($handle)
  if ($ok) { return $builder.ToString() }
  return ''
}

$regValue = Get-ItemProperty -Path $lmKey -ErrorAction SilentlyContinue
if ($null -eq $regValue -or [string]::IsNullOrEmpty($regValue.WeaselRoot)) {
  throw 'WeaselRoot is not registered in HKLM Software\Rime\Weasel'
}
$root = $regValue.WeaselRoot
$expectedServer = Join-Path $root 'WeaselServer.exe'
if (-not (Test-Path -LiteralPath $expectedServer)) {
  throw "Expected server not found: $expectedServer"
}

$clsid = '{A3F4CDED-B1E9-41EE-9CA6-7B4D0DE6CB0A}'
$tsfKey = 'HKLM:\SOFTWARE\Classes\CLSID\' + $clsid + '\InprocServer32'
$expectedTsf = 'E:\codex_data\研究\weasel-baseline\output\v0.3-tsf-runtime12\weaselx64.dll'
$tsfValue = (Get-ItemProperty -Path $tsfKey -ErrorAction SilentlyContinue).'(default)'
if ($tsfValue -ne $expectedTsf) {
  throw "TSF CLSID points to wrong DLL: $tsfValue (expected $expectedTsf). Run output\register-tsf-v0.3.ps1 as admin."
}

$userDir = Join-Path $env:APPDATA 'Rime'
New-Item -ItemType Directory -Force -Path $userDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $userDir 'build') | Out-Null
Set-Content -LiteralPath (Join-Path $userDir 'weasel-exp-version.txt') -Value $expectedVersion -Encoding UTF8

if (-not (Test-Path $rimeKey)) { New-Item -Path $rimeKey -Force | Out-Null }
$currentUserDir = (Get-ItemProperty -Path $rimeKey -Name RimeUserDir -ErrorAction SilentlyContinue).RimeUserDir
if ($currentUserDir -ne $userDir) {
  New-ItemProperty -Path $rimeKey -Name RimeUserDir -Value $userDir -PropertyType String -Force | Out-Null
}

$patchedSchema = Join-Path $root 'RimeUser\build\rime_ice.schema.yaml'
$activeSchema = Join-Path $userDir 'build\rime_ice.schema.yaml'
if ((Test-Path -LiteralPath $patchedSchema) -and
    ((Get-Content -LiteralPath $activeSchema -Raw -ErrorAction SilentlyContinue) -match 'reset:\s*1')) {
  Copy-Item -LiteralPath $patchedSchema -Destination $activeSchema -Force
}

$matching = $false
foreach ($process in Get-Process -Name WeaselServer -ErrorAction SilentlyContinue) {
  $image = Get-ProcessImageName $process.Id
  if ([string]::IsNullOrEmpty($image)) { continue }
  if ([string]::Equals($image, $expectedServer, [StringComparison]::OrdinalIgnoreCase)) {
    $matching = $true
  } else {
    Write-Host "[guard] kill wrong WeaselServer: $image"
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
  }
}

if (-not $matching) {
  Write-Host "[guard] start $expectedServer"
  Start-Process -FilePath $expectedServer -WindowStyle Hidden
  Start-Sleep -Milliseconds 800
}
Write-Host "[guard] version=$expectedVersion"
Write-Host "[guard] root=$root"
Write-Host "[guard] user=$userDir"