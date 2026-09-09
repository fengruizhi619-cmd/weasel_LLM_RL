<#
.SYNOPSIS
    Install (or remove) the weasel_LLM_RL build into an existing Rime/Weasel install.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -RimeHome "C:\Program Files\Rime\weasel-0.17.4"
    .\install.ps1 -Uninstall
    .\install.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$RimeHome = "",
    [string]$Source = "",
    [switch]$Uninstall,
    [switch]$SkipRegistry,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ClsidKey = 'HKLM:\SOFTWARE\Classes\CLSID\{A3F4CDED-B1E9-41EE-9CA6-7B4D0DE6CB0A}\InprocServer32'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Files = @('weaselx64.dll', 'WeaselServer.exe')
$Scripts = @('ghost_service.cmd', 'ghost_data_panel.cmd')

function Say([string]$m) { Write-Host $m }
function Step([string]$m) { Write-Host ("  " + $m) }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Find-RimeHome {
    if ($RimeHome) { return $RimeHome }
    try {
        $dll = (Get-ItemProperty -LiteralPath $ClsidKey -ErrorAction Stop).'(default)'
        if ($dll -and (Test-Path (Join-Path (Split-Path -Parent $dll) 'WeaselServer.exe'))) {
            return (Split-Path -Parent $dll)
        }
    } catch {}
    foreach ($root in @('C:\Program Files\Rime', 'C:\Program Files (x86)\Rime')) {
        if (-not (Test-Path $root)) { continue }
        $hit = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
               Where-Object { Test-Path (Join-Path $_.FullName 'WeaselServer.exe') } |
               Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

function Find-Source {
    if ($Source) { return $Source }
    foreach ($d in @('bin', 'output')) {
        $p = Join-Path $RepoRoot $d
        if ((Test-Path (Join-Path $p 'weaselx64.dll')) -and
            (Test-Path (Join-Path $p 'WeaselServer.exe'))) { return $p }
    }
    return $null
}

function Stop-Weasel {
    $p = Get-Process WeaselServer -ErrorAction SilentlyContinue
    if ($p) { $p | Stop-Process -Force; Start-Sleep -Milliseconds 1500 }
}

function Start-Weasel([string]$RimeDir) {
    $exe = Join-Path $RimeDir 'WeaselServer.exe'
    if (Test-Path $exe) { Start-Process -FilePath $exe -WindowStyle Hidden }
}

# --- elevate unless we are only previewing ---
if (-not $DryRun -and -not (Test-Admin)) {
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"")
    if ($RimeHome) { $argList += @('-RimeHome', "`"$RimeHome`"") }
    if ($Source)   { $argList += @('-Source', "`"$Source`"") }
    if ($Uninstall) { $argList += '-Uninstall' }
    if ($SkipRegistry) { $argList += '-SkipRegistry' }
    Start-Process powershell -Verb RunAs -ArgumentList $argList
    exit 0
}

$RimeDir = Find-RimeHome
if (-not $RimeDir) { throw "找不到 Rime 安装目录，请用 -RimeHome 指定" }
if (-not (Test-Path $RimeDir)) { throw "目录不存在: $RimeDir" }
Say "Rime 目录: $RimeDir"

if ($Uninstall) {
    Say "卸载 weasel_LLM_RL"
    foreach ($f in $Files) {
        $cand = Get-ChildItem (Join-Path $RimeDir ($f + '.*.bak')) -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($cand) {
            Step ("恢复 " + $f + " <- " + $cand.Name)
            if (-not $DryRun) {
                Stop-Weasel
                Copy-Item -LiteralPath $cand.FullName -Destination (Join-Path $RimeDir $f) -Force
            }
        } else {
            Step ("没有 " + $f + " 的备份，跳过")
        }
    }

    foreach ($s in $Scripts + @('ghost_home.txt')) {
        $sp = Join-Path $RimeDir $s
        if (Test-Path $sp) {
            Step ("删除 " + $s)
            if (-not $DryRun) { [IO.File]::Delete($sp) }
        }
    }
    if (-not $SkipRegistry) {
        $dll = Join-Path $RimeDir 'weaselx64.dll'
        Step ("注册表 -> " + $dll)
        if (-not $DryRun) { & reg.exe add "HKLM\SOFTWARE\Classes\CLSID\{A3F4CDED-B1E9-41EE-9CA6-7B4D0DE6CB0A}\InprocServer32" /ve /d "$dll" /f | Out-Null }
    }
    if (-not $DryRun) { Start-Weasel $RimeDir }
    Say "卸载完成。"
    exit 0
}

$src = Find-Source
if (-not $src) { throw "找不到编译产物（bin/ 或 output/ 下的 weaselx64.dll + WeaselServer.exe）" }
Say "产物目录: $src"

foreach ($f in $Files) {
    if (-not (Test-Path (Join-Path $src $f))) { throw ("缺少 " + $f + "：" + $src) }
}
foreach ($s in $Scripts) {
    $p = Join-Path $RepoRoot ("tools\LlamaTreeExp\" + $s)
    if (-not (Test-Path $p)) { throw ("缺少 " + $s + "：" + $p) }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'

Say "1/6 停止 WeaselServer"
if (-not $DryRun) { Stop-Weasel }

Say "2/6 备份现有文件"
foreach ($f in $Files) {
    $t = Join-Path $RimeDir $f
    if (Test-Path $t) {
        Step ($f + " -> " + $f + "." + $stamp + ".bak")
        if (-not $DryRun) { Copy-Item -LiteralPath $t -Destination ($t + "." + $stamp + ".bak") -Force }
    }
}

Say "3/6 复制 dll / exe"
foreach ($f in $Files) {
    Step $f
    if (-not $DryRun) { Copy-Item -LiteralPath (Join-Path $src $f) -Destination (Join-Path $RimeDir $f) -Force }
}

Say "4/6 复制 ghost 脚本"
foreach ($s in $Scripts) {
    Step $s
    if (-not $DryRun) {
        Copy-Item -LiteralPath (Join-Path $RepoRoot ("tools\LlamaTreeExp\" + $s)) -Destination (Join-Path $RimeDir $s) -Force
    }
}

Say "5/6 写入 ghost_home.txt"
Step $RepoRoot
if (-not $DryRun) {
    [IO.File]::WriteAllText((Join-Path $RimeDir 'ghost_home.txt'), $RepoRoot + "`r`n",
                            [Text.Encoding]::GetEncoding(936))
}

if (-not $SkipRegistry) {
    Say "6/6 注册 TSF CLSID"
    $dll = Join-Path $RimeDir 'weaselx64.dll'
    Step $dll
    if (-not $DryRun) { & reg.exe add "HKLM\SOFTWARE\Classes\CLSID\{A3F4CDED-B1E9-41EE-9CA6-7B4D0DE6CB0A}\InprocServer32" /ve /d "$dll" /f | Out-Null }
} else {
    Say "6/6 跳过注册表"
}

if (-not $DryRun) { Start-Weasel $RimeDir }

Say ""
Say "安装完成。接下来："
Say ("  1. 把 Qwen3-0.6B-Base 放到 " + $RepoRoot + "\models\Qwen3-0.6B-Base")
Say "     （或设置环境变量 WEASEL_LLM_MODEL 指向别处）"
Say "  2. 在任意输入框打字验证；模式切换见 README"
