# register-task.ps1 - 注册无头看门狗计划任务
#
# 必须在【管理员 PowerShell】里运行一次（任务文件在 System32\Tasks 下，普通会话改不动）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File "E:\DSH_data\研究\weasel-baseline\output\register-task.ps1"
#
# 只想检查、不改系统时加 -DryRun：
#   powershell -NoProfile -ExecutionPolicy Bypass -File "...\register-task.ps1" -DryRun
#
# 关键点：任务动作是 wscript.exe 调 watchdog.vbs（无头启动器），**不是** powershell.exe。
# 直接把 powershell.exe 当任务动作，即使加了 -WindowStyle Hidden，Windows 也会先为它
# 创建控制台窗口、随后才隐藏 —— 表现就是每 5 分钟闪一次窗口。见 watchdog.vbs 顶部注释。
#
# 历史坑（2026-09-12 踩过，别再用 [TimeSpan]::MaxValue）：
#   RepetitionDuration 若超出任务计划允许的范围，生成的是 P99999999DT23H59M59S，
#   注册直接失败："任务 XML 包含格式不正确或超出范围的值 (8,42):Duration:P99999999DT23H59M59S"。
#   这里固定用 31 天（每 5 分钟重复，31 天后由 -StartWhenAvailable 与登录触发续上）。

param(
  [switch]$DryRun
)

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

# 31 天：确定在任务计划允许范围内（上限以内），且明显管用
$repetitionDuration = New-TimeSpan -Days 31

$action    = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument ('//B //NoLogo "' + $vbs + '"')
$principal = New-ScheduledTaskPrincipal -UserId ("{0}\{1}" -f $env:USERDOMAIN, $env:USERNAME) -LogonType Interactive -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

$triggers = @(
  (New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration $repetitionDuration),
  (New-ScheduledTaskTrigger -AtLogOn)
)

# 硬断言：动作参数必须真的带上 vbs 路径。
# New-ScheduledTaskAction 在参数丢失/类型不对时不会报错，只会返回空 Arguments —— 那样注册出来的
# 任务等于跑一个没有参数的 wscript.exe，看门狗彻底失效，而且不报任何错。这里宁可当场失败。
if (-not $action.Arguments -or $action.Arguments -notlike '*watchdog.vbs*') {
  Note ("FAIL 动作参数异常，Arguments=[" + $action.Arguments + "]")
  throw ("动作参数异常：wscript 没拿到 watchdog.vbs 路径（Arguments=[" + $action.Arguments + "]），拒绝注册")
}

# 预检：把将要注册的内容打印出来，确认 Duration/Interval 合法（-DryRun 只做这一步）
$t0 = $triggers[0]
$dur = $t0.Repetition.Duration
$ivl = $t0.Repetition.Interval

Write-Host ("动作      : {0} {1}" -f $action.Execute, $action.Arguments)
Write-Host ("重复      : Interval={0}  Duration={1}" -f $ivl, $dur)
Note ("check: action={0} {1} interval={2} duration={3}" -f $action.Execute, $action.Arguments, $ivl, $dur)

if ($DryRun) {
  Write-Host ''
  Write-Host '-DryRun：只检查，未注册任何任务。'
  exit 0
}

$results = @()
foreach ($name in 'WeaselOnlineWatchdog', 'WeaselOnlineWatchdogLogon') {
  try {
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Principal $principal -Settings $settings -Force | Out-Null
    $results += "$name=OK"
  } catch {
    $results += ("$name=FAIL(" + $_.Exception.Message.Trim() + ")")
  }
}
Note ($results -join ' | ')
Write-Host ($results -join ' | ')

# 回读确认：动作必须指向 wscript + watchdog.vbs
foreach ($name in 'WeaselOnlineWatchdog', 'WeaselOnlineWatchdogLogon') {
  try {
    $t = Get-ScheduledTask -TaskName $name -ErrorAction Stop
    $a = $t.Actions[0]
    $i = $t | Get-ScheduledTaskInfo
    $line = "{0}: {1} {2}   (NextRun={3})" -f $name, $a.Execute, $a.Arguments, $i.NextRunTime
    Note ("verify " + $line)
    Write-Host $line
  } catch {
    Note ("verify $name FAIL: " + $_.Exception.Message)
  }
}

Write-Host ''
Write-Host '完成。动作应是 wscript.exe + watchdog.vbs（无头）。'
Write-Host ('日志：' + $log)
