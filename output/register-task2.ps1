# register-task2.ps1 - 兼容入口（旧名）
#
# 注册逻辑已统一到 register-task.ps1，本文件只做转发，避免两处各写一套、
# 再次出现"任务动作指向 powershell.exe 导致每 5 分钟闪窗"的问题。
#
# 必须在【管理员 PowerShell】里运行：
#   powershell -NoProfile -ExecutionPolicy Bypass -File "<repo>\output\register-task.ps1"

$ErrorActionPreference = 'Stop'
$main = Join-Path $PSScriptRoot 'register-task.ps1'
if (-not (Test-Path $main)) { throw "找不到 register-task.ps1: $main" }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $main
exit $LASTEXITCODE
