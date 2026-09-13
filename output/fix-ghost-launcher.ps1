# fix-ghost-launcher.ps1 - 修复数据面板/引擎启动器（把仓库路径直接写进安装目录的 .cmd）
#
# 必须在【管理员 PowerShell】里运行：
#   powershell -NoProfile -ExecutionPolicy Bypass -File "E:\DSH_data\研究\weasel-baseline\output\fix-ghost-launcher.ps1"
#
# 只检查不改动（任何会话都能跑）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File "...\fix-ghost-launcher.ps1" -DryRun
#
# 为什么需要它
#   安装目录里的 ghost_data_panel.cmd / ghost_service.cmd 仍从 ghost_home.txt 读仓库路径
#   （`set /p` 按 ANSI 代码页读）。只要那个 txt 被任何工具用 UTF-8 写过一次，路径就变
#   乱码（研究 -> 鐮旂┒），pythonw 找不到脚本，面板静默起不来、且没有任何报错。
#   本脚本按 install.ps1 第 4 步的口径重新生成这两个 .cmd：把仓库路径作为 ASCII 字面量
#   写进去，从此不再依赖那个 txt 的编码。
#
# 不做什么：不复制 weaselx64.dll / WeaselServer.exe（它们已被 explorer 等进程以 TSF 形式
#   加载，且与产物同哈希，无需替换），不停也不重启 WeaselServer。

param(
    [switch]$DryRun,
    [switch]$Launch
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot          # output\ -> 仓库根
$inst = 'C:\Program Files\Rime\weasel-0.17.4-emoji-off'
if (-not (Test-Path $inst)) {
    $dll = (Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Classes\CLSID\{A3F4CDED-B1E9-41EE-9CA6-7B4D0DE6CB0A}\InprocServer32' -ErrorAction SilentlyContinue).'(default)'
    if ($dll) { $inst = Split-Path -Parent $dll }
}
if (-not (Test-Path $inst)) { throw '找不到 Rime 安装目录' }

$gbk = [Text.Encoding]::GetEncoding(936)

Write-Host "仓库根  : $repo"
Write-Host "安装目录: $inst"

# 现状诊断：安装目录里的 .cmd 现在是不是还在依赖 ghost_home.txt
foreach ($name in 'ghost_service.cmd', 'ghost_data_panel.cmd') {
    $p = Join-Path $inst $name
    if (Test-Path $p) {
        $cur = $gbk.GetString([IO.File]::ReadAllBytes($p))
        $stale = $cur.Contains('set /p GHOST_HOME')
        $ok = $cur.Contains('set "GHOST_HOME=' + $repo + '"')
        Write-Host ("  现状 {0,-22} 仍读 ghost_home.txt={1}  已内嵌正确路径={2}" -f $name, $stale, $ok)
    } else {
        Write-Host ("  现状 {0,-22} 不存在" -f $name)
    }
}
$homeFile = Join-Path $inst 'ghost_home.txt'
if (Test-Path $homeFile) {
    $ansi = $gbk.GetString([IO.File]::ReadAllBytes($homeFile)).Trim()
    $utf8 = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($homeFile)).Trim()
    Write-Host ("  ghost_home.txt 按ANSI='{0}'  按UTF8='{1}'" -f $ansi, $utf8)
}

if ($DryRun) {
    Write-Host ''
    Write-Host '-DryRun：只诊断，未改动任何文件。'
    exit 0
}

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$admin = (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw '需要管理员权限（要写 Program Files）' }

foreach ($name in 'ghost_service.cmd', 'ghost_data_panel.cmd') {
    $tpl = Get-Content -LiteralPath (Join-Path $repo ("tools\LlamaTreeExp\" + $name)) -Raw
    if (-not $tpl.Contains('@@GHOST_HOME@@')) { throw ('模板缺少 @@GHOST_HOME@@ 占位：' + $name) }
    $body = $tpl.Replace('@@GHOST_HOME@@', $repo)
    $body = ($body -replace "`r`n", "`n") -replace "`n", "`r`n"   # 统一 CRLF
    [IO.File]::WriteAllText((Join-Path $inst $name), $body, $gbk)

    # 回读校验
    $read = [IO.File]::ReadAllText((Join-Path $inst $name), $gbk)
    if ($read.Contains('@@GHOST_HOME@@')) { throw ($name + ' 占位未被替换') }
    if (-not $read.Contains($repo)) { throw ($name + ' 里的仓库路径不正确') }
    $line = ($read -split "`r`n" | Where-Object { $_ -match 'set "GHOST_HOME=' } | Select-Object -First 1).Trim()
    Write-Host ('  [OK] {0,-22} {1}' -f $name, $line)
}

# 旧版回退文件：仍然写 GBK，这样万一还有旧 .cmd 在用也能正确读出路径
[IO.File]::WriteAllText($homeFile, $repo + "`r`n", $gbk)
$ansi2 = $gbk.GetString([IO.File]::ReadAllBytes($homeFile)).Trim()
Write-Host ('  [OK] ghost_home.txt      按 ANSI 读出: {0}' -f $ansi2)
if ($ansi2 -ne $repo) { throw 'ghost_home.txt 编码仍不正确' }

Write-Host ''
Write-Host '完成。现在点语言栏的「数据面板」，窗口应能弹出。'

if ($Launch) {
    Write-Host '正在直接以安装目录的 .cmd 起一次面板（用于当场验收）...'
    & cmd.exe /c ('start "LLM Data Panel" "' + (Join-Path $inst 'ghost_data_panel.cmd') + '"')
    Start-Sleep -Seconds 5
    $w = Get-Process -EA SilentlyContinue | Where-Object { $_.MainWindowTitle -match 'LLM 数据面板' }
    if ($w) { Write-Host ('  [OK] 面板窗口已出现：PID=' + $w.Id + ' 「' + $w.MainWindowTitle + '」') }
    else { Write-Host '  [FAIL] 未检测到面板窗口，请把上面输出发回' }
}
