' watchdog.vbs - headless launcher for the scheduled task
'
' WHY THIS FILE EXISTS
' If the scheduled task action is powershell.exe directly, Windows still creates a
' console window for it first and only hides it afterwards (-WindowStyle Hidden is
' too late) -- the symptom is a window flashing every 5 minutes.
' Started through wscript.exe + WshShell.Run with windowStyle=0, no console is ever
' created: wscript itself is a windowless host, and the child powershell is spawned
' with SW_HIDE from the start.
'
' Scheduled task action must be:
'   wscript.exe //B //NoLogo "<repo>\tools\LlamaTreeExp\watchdog.vbs"
' Register it with: output\register-task.ps1  (run once, from an elevated PowerShell)
'
' This file is intentionally pure ASCII with no BOM: wscript reads .vbs as ANSI,
' so non-ASCII bytes here would be mis-decoded and could corrupt parsing.

Option Explicit

Dim fso, sh, here, ps1, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

here = fso.GetParentFolderName(WScript.ScriptFullName)
ps1  = fso.BuildPath(here, "watchdog.ps1")

' If the payload script is missing, exit silently - never raise a dialog.
If Not fso.FileExists(ps1) Then WScript.Quit 0

cmd = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """"

On Error Resume Next
sh.Run cmd, 0, False
If Err.Number <> 0 Then Err.Clear
On Error GoTo 0

WScript.Quit 0
