cli_emojiless_RL_v2.0.2  (2026-09-12)

只动看门狗与路径：训练核心、推理引擎、权重一个字没碰。

v2.0.1 -> v2.0.2 增量

  1. 看门狗改为完全无头（修每 5 分钟闪窗）
     - 症状：计划任务 WeaselOnlineWatchdog 每 5 分钟闪一次窗口
     - 根因：任务动作直接是 powershell.exe。它是控制台程序，由计划任务以交互身份
       （LogonType=InteractiveToken）启动时，Windows 会先分配控制台窗口，而
       -WindowStyle Hidden 是进程起来之后才生效的 —— 那一瞬间的窗口会露出来
     - 修法：动作改为 wscript.exe 调 tools\LlamaTreeExp\watchdog.vbs，由 WshShell.Run
       的 windowStyle=0 拉起 powershell（SW_HIDE，从一开始就不创建控制台）。
       wscript 自身是无窗口宿主
     - watchdog.vbs 重写为纯 ASCII、无 BOM（wscript 按 ANSI 读 .vbs，非 ASCII 会被误解码），
       路径用 fso.GetParentFolderName(WScript.ScriptFullName) 自推、不硬编码；
       payload 缺失时静默退出，不弹错误框

  2. 注册脚本 output\register-task.ps1 修三处
     - RepetitionDuration 用 [TimeSpan]::MaxValue 会生成越界的
       <Duration>P99999999DT23H59M59S</Duration>，任务计划直接拒收
       （"任务 XML 包含格式不正确或超出范围的值 (8,42):Duration:..."），
       导致无头改造一直没生效。现固定 31 天（P31D），配合 -StartWhenAvailable 与登录触发续期
     - 新增硬断言：动作参数里必须出现 watchdog.vbs，否则当场失败。
       起因：New-ScheduledTaskAction 参数异常时不报错、只返回空 Arguments，
       那种任务注册出来等于跑一个没参数的 wscript.exe，看门狗静默失效、且没有任何报错
     - 新增 -DryRun：只构造并打印将要注册的动作与重复参数，不碰任何系统任务，
       普通（非管理员）会话也能自检。注册前后把 check / 动作 / NextRun 写进 register-task.log
     - register-task2.ps1 改为转发到 register-task.ps1，避免两处各写一套注册逻辑
     - register-task.ps1 存为 UTF-8 with BOM：否则 PS 5.1 按 GBK 解析，中文路径会变乱码

  3. 路径：codex -> DSH
     - 工作区从 E:\codex_data\研究 迁到 E:\DSH_data\研究 后残留的硬路径改为新位置：
       experiments/backbone_compare.py（MODELS 两个基线权重目录）、
       experiments/backbone_head_train.py（BASE / CHAT）、
       output/register-task.ps1 与 register-task2.ps1（计划任务日志与看门狗路径）
     - 改前这些脚本引用的是已不存在的 E:\codex_data\...，在本机跑不起来
     - 另修复 weasel-baseline 工作树与仓库的 git 连接（gitdir 指向旧的 codex 路径）

  不含：训练核心（损失 / 优化器 / 学习率 / 混料）与全部权重文件，与 v2.0.1 相同。

生效条件（重要）
  - 无头看门狗需要动计划任务，普通会话改不动（任务文件在 System32\Tasks 下）。
    在【管理员 PowerShell】里跑一次：
      powershell -NoProfile -ExecutionPolicy Bypass -File "<repo>\output\register-task.ps1"
  - 不重注册的话，任务动作仍是 powershell.exe，闪窗照旧。

验收方法
  - 动作回读：Get-ScheduledTask -TaskName WeaselOnlineWatchdog | % { $_.Actions[0].Execute + ' ' + $_.Actions[0].Arguments }
    应为 wscript.exe //B //NoLogo "...\tools\LlamaTreeExp\watchdog.vbs"
  - 之后每 5 分钟：窗口不再出现，同时 tools\LlamaTreeExp\diag\watchdog.log 仍在按时新增行
    （两者必须同时成立：只不闪不写日志 = 看门狗死了）
