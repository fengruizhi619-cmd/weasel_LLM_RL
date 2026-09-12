# register-task.ps1 - 注册无头看门狗计划任务
#
# 必须在【管理员 PowerShell】里运行一次（任务文件在 System32\Tasks 下，普通会话改不动）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File "E:\DSH_data\研究\weasel-baseline\output\register-task.ps1"
#
# 关键点：任务动作是 wscript.exe 调 watchdog.vbs（无头启动器），**不是** powershell.exe。
# 直接把 powershell.exe 当任务动作，即使加了 -WindowStyle Hidden，Windows 也会先为它
# 创建控制台窗口、随后才隐藏 —— 表现就是每 5 分钟闪一次窗口。见 watchdog.vbs 顶部注释。

$ErrorActionPreference = 'Stop'

# 从脚本位置推仓库根：output\ -> weasel-baseline\
$root = Split-Path -Parent $PSScriptRoot
$vbs  = Join-Path $root 'tools\LlamaTreeExp\watchdog.vbs'
$ps1  = Join-Path $root 'tools\LlamaTreeExp\watchdog.ps1'
$log  = Join-Path $PSScriptRoot 'register-task.log'

function Note([string]$m) {
  Add-Content -LiteralPath $log -Value ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $m) -Encoding UTF8
}

if (-not (Test-Path $vbs)) { Note "FAIL 找不到 watchdog.vbs: $vbs"; throw "找不到 watchdog.vbs: $vbs" }
if (-not (Test-Path $ps1)) { Note "FAIL 找不到 watchdog.ps1: $ps1"; throw "找不到 watchdog.ps1: $ps1" }

$action    = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument ('//B //NoLogo "' + $vbs + '"')
$principal = New-ScheduledTaskPrincipal -UserId ("{0}\{1}" -f $env:USERDOMAIN, $env:USERNAME) -LogonType Interactive -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$triggers  = @(
  (New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration ([TimeSpan]::MaxValue)),
  (New-ScheduledTaskTrigger -AtLogOn)
)

$results = @()
foreach ($name in 'WeaselOnlineWatchdog', 'WeaselOnlineWatchdogLogon') {
  try {
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Principal $principal -Settings $settings -Force | Out-Null
    $results += "$name=OK"
  } catch {
    $results += ("$name=FAIL(" + $_.Exception.Message + ")")
  }
}
Note ($results -join ' | ')

# 回读确认：动作必须指向 wscript + watchdog.vbs
foreach ($name in 'WeaselOnlineWatchdog', 'WeaselOnlineWatchdogLogon') {
  try {
    $t = Get-ScheduledTask -TaskName $name -ErrorAction Stop
    $a = $t.Actions[0]
    Note ("verify {0}: {1} {2}" -f $name, $a.Execute, $a.Arguments)
    Write-Host ("{0}: {1} {2}" -f $name, $a.Execute, $a.Arguments)
  } catch {
    Note ("verify $name FAIL: " + $_.Exception.Message)
  }
}

Write-Host ''
Write-Host '完成。动作应是 wscript.exe + watchdog.vbs（无头）。'
Write-Host ('日志：' + $log)
