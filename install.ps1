<#
.SYNOPSIS
    Install (or remove) the weasel_LLM_RL build into an existing Rime/Weasel install.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -RimeHome "C:\Program Files\Rime\weasel-0.17.4"
    .\install.ps1 -Uninstall
    .\install.ps1 -DryRun
    .\install.ps1 -DownloadModels
    .\install.ps1 -ModelsOnly
    .\install.ps1 -ModelsOnly -Mirror https://gh.xxooo.cf/
#>
[CmdletBinding()]
param(
    [string]$RimeHome = "",
    [string]$Source = "",
    [switch]$Uninstall,
    [switch]$SkipRegistry,
    [switch]$DryRun,
    [switch]$DownloadModels,
    [switch]$ModelsOnly,
    [string]$Mirror = ""
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

$ReleaseTag  = 'cli_emojiless_RL_v2.0'
$ReleaseBase = 'https://github.com/fengruizhi619-cmd/weasel_LLM_RL/releases/download/' + $ReleaseTag + '/'
$ModelParts  = @('Qwen3-0.6B-Base.zip.001', 'Qwen3-0.6B-Base.zip.002', 'Qwen3-0.6B-Base.zip.003')
$HeadAsset   = 'lm_head_t0.pt'

function Get-MirrorBase {
    if ($Mirror) { return $Mirror }
    if ($env:WEASEL_LLM_MIRROR) { return $env:WEASEL_LLM_MIRROR }
    return ''
}

function Get-RemoteFile([string]$Name, [string]$Dest) {
    $url = $ReleaseBase + $Name
    $mb = Get-MirrorBase
    if ($mb) {
        if (-not $mb.EndsWith('/')) { $mb += '/' }
        $url = $mb + $url
    }
    Step ("下载 " + $Name)
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L --fail --retry 3 --retry-delay 3 -o "$Dest" "$url" -w "    http=%{http_code} size=%{size_download} speed=%{speed_download}`n"
        if ($LASTEXITCODE -ne 0) { throw ("下载失败: " + $url) }
    } else {
        $old = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        try { Invoke-WebRequest -Uri $url -OutFile $Dest -UseBasicParsing }
        finally { $ProgressPreference = $old }
    }
}

function Install-Models {
    $modelDir = Join-Path $RepoRoot 'models\Qwen3-0.6B-Base'
    $headDir  = Join-Path $RepoRoot 'tools\LlamaTreeExp\diag\checkpoints_online'
    $tmp      = Join-Path $RepoRoot 'models\_download'
    foreach ($d in @($modelDir, $headDir, $tmp)) {
        if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
    }

    if (Test-Path (Join-Path $modelDir 'model.safetensors')) {
        Step "主干已存在，跳过"
    } else {
        foreach ($part in $ModelParts) { Get-RemoteFile $part (Join-Path $tmp $part) }
        $zip = Join-Path $tmp 'Qwen3-0.6B-Base.zip'
        Step "合并分段"
        $out = [IO.File]::Create($zip)
        try {
            foreach ($part in $ModelParts) {
                $in = [IO.File]::OpenRead((Join-Path $tmp $part))
                try { $in.CopyTo($out) } finally { $in.Close() }
            }
        } finally { $out.Close() }
        Step "解压到 models\Qwen3-0.6B-Base"
        Expand-Archive -LiteralPath $zip -DestinationPath $modelDir -Force
        Remove-Item -LiteralPath $tmp -Recurse -Force
    }

    $head = Join-Path $headDir $HeadAsset
    if (Test-Path $head) {
        Step "解码器已存在，跳过"
    } else {
        Get-RemoteFile $HeadAsset $head
    }
}

# --- elevate unless we are only previewing ---
if (-not $DryRun -and -not $ModelsOnly -and -not (Test-Admin)) {
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"")
    if ($RimeHome) { $argList += @('-RimeHome', "`"$RimeHome`"") }
    if ($Source)   { $argList += @('-Source', "`"$Source`"") }
    if ($Uninstall) { $argList += '-Uninstall' }
    if ($SkipRegistry) { $argList += '-SkipRegistry' }
    if ($DownloadModels) { $argList += '-DownloadModels' }
    if ($ModelsOnly) { $argList += '-ModelsOnly' }
    if ($Mirror) { $argList += @('-Mirror', "`"$Mirror`"") }
    Start-Process powershell -Verb RunAs -ArgumentList $argList
    exit 0
}

if ($ModelsOnly) {
    Say "仅下载模型（不安装输入法）"
    if (-not $DryRun) { Install-Models }
    Say "完成。"
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

if ($DownloadModels) {
    Say "7/7 下载模型（主干 + 解码器，约 1.8GB）"
    if (-not $DryRun) { Install-Models }
}

if (-not $DryRun) { Start-Weasel $RimeDir }

Say ""
Say "安装完成。接下来："
if (Test-Path (Join-Path $RepoRoot 'models\Qwen3-0.6B-Base\model.safetensors')) {
    Say "  - 主干已就位: models\Qwen3-0.6B-Base"
} else {
    Say "  - 主干未就位，运行 .\install.ps1 -ModelsOnly 下载（约 1.8GB）"
    Say ("    或手动放到 " + $RepoRoot + "\models\Qwen3-0.6B-Base")
}
if (Test-Path (Join-Path $RepoRoot 'tools\LlamaTreeExp\diag\checkpoints_online\lm_head_t0.pt')) {
    Say "  - 解码器已就位: tools\LlamaTreeExp\diag\checkpoints_online\lm_head_t0.pt"
} else {
    Say "  - 解码器未就位，运行 .\install.ps1 -ModelsOnly 下载"
}
Say "  - 在任意输入框打字验证；模式切换见 README"
Say "  - 下载慢时可加 -Mirror https://gh.xxooo.cf/ 走镜像"
