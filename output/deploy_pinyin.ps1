$ErrorActionPreference = 'Stop'
$outDir = $PSScriptRoot
$log    = Join-Path $outDir 'deploy_pinyin.log'
function Say($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
  Write-Host $line
  Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

$src = Join-Path $outDir 'weaselx64.dll'
$dst = 'C:\Program Files\Rime\weasel-0.17.4-emoji-off\weaselx64.dll'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Say ("admin = " + $isAdmin)
Say ("src = " + $src + " (" + (Get-Item -LiteralPath $src).Length + " bytes)")
Say ("dst = " + $dst + " (" + (Get-Item -LiteralPath $dst).Length + " bytes)")

Say "1/4 stop WeaselServer"
Get-Process WeaselServer -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 1500

Say "2/4 move old dll aside"
$bak = "$dst.bak-$stamp"
Move-Item -LiteralPath $dst -Destination $bak -Force
Say ("moved -> " + $bak)

Say "3/4 copy new dll"
Copy-Item -LiteralPath $src -Destination $dst -Force
Say ("new size = " + (Get-Item -LiteralPath $dst).Length)

Say "4/4 start WeaselServer"
Start-Process -FilePath 'C:\Program Files\Rime\weasel-0.17.4-emoji-off\WeaselServer.exe' -WindowStyle Hidden
Start-Sleep -Seconds 3
$p = Get-Process WeaselServer -ErrorAction SilentlyContinue
if ($p) { Say ("WeaselServer pid = " + $p.Id) } else { Say "WeaselServer NOT running" }
Say "DONE"
