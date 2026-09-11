$ErrorActionPreference = 'Continue'
$log = 'E:\codex_data\研究\weasel-baseline\output\register-task.log'
$wd = 'E:\codex_data\研究\weasel-baseline\tools\LlamaTreeExp\watchdog.ps1'
$tr = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $wd + '"'
$out1 = schtasks /Create /TN "WeaselOnlineWatchdog" /TR $tr /SC MINUTE /MO 5 /IT /F 2>&1
$out2 = schtasks /Create /TN "WeaselOnlineWatchdogLogon" /TR $tr /SC ONLOGON /IT /F 2>&1
Add-Content -LiteralPath $log -Value ((Get-Date -Format 'HH:mm:ss') + ' 5min=' + ($out1 -join ' ') + ' | logon=' + ($out2 -join ' ')) -Encoding UTF8