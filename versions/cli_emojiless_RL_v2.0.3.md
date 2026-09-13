cli_emojiless_RL_v2.0.3  (2026-09-13)

修复"数据面板打不开"与安装脚本的两处脆弱点：训练核心 / 推理引擎 / 权重未动。

背景（本次故障的真实成因）
  安装目录里的 ghost_data_panel.cmd / ghost_service.cmd 用

      if exist "%~dp0ghost_home.txt" set /p GHOST_HOME=<"%~dp0ghost_home.txt"

  从 ghost_home.txt 读仓库根。cmd 的 `set /p` 按 **ANSI 代码页**读文件，而该文件被
  写成了 **UTF-8**：

      字节  45 3A 5C … 5C | E7 A0 94 E7 A9 B6 | 5C …      <- 研究 的 UTF-8
      正确应为 D1 D0 BE BF                                  <- 研究 的 GBK

  于是 cmd 拿到的路径是 `E:\DSH_data\鐮旂┒\weasel-baseline`，pythonw 找不到脚本，
  面板**静默起不来、没有任何报错**。同一文件也供在线引擎使用，所以切回 online 模式
  同样会起不来。
  （注：install.ps1 第 245 行本来就写 GBK，是后来被别的东西用 UTF-8 覆盖了——
    2026-09-13 15:47:58 那次写入即为此。）

v2.0.2 -> v2.0.3 增量

  1. 启动器不再依赖 ghost_home.txt 的编码（根因消除，而非纠正内容）
     - ghost_service.cmd / ghost_data_panel.cmd 改为在安装期把仓库路径**以 ASCII 字面量
       写进脚本本身**：模板里留 @@GHOST_HOME@@ 占位，install.ps1 第 4 步替换后落盘。
       路径不再经过任何"按代码页解析的文本文件"，编码问题从设计上消失。
     - 两个 .cmd 保持纯 ASCII、无 BOM、CRLF（非 ASCII 字节 = 0，已校验）。
     - 仓库里保留占位符版本，不提交替换后的副本。
     - 回退顺序：内嵌路径 -> %WEASEL_LLM_HOME% -> %~dp0..\..（旧安装仍可用）。

  2. install.ps1 第 3 步：内容一致就跳过，不再因 dll 被占用而中断
     - 症状：`Copy-Item : 文件 weaselx64.dll 正由另一进程使用`。
       占用者是 explorer / DSH Desktop / SearchHost 等——weaselx64.dll 是 TSF 进程内组件，
       只停 WeaselServer 释放不了它。
     - 之前硬拷会抛异常并中断整条安装，导致第 4/5/6 步（生成脚本、写 ghost_home.txt、
       注册 CLSID）全部跑不到——本次故障就是这么被卡住的。
     - 现在先比 SHA256：一致则打印"已是同一份，跳过"继续往下走。

  3. install.ps1 第 4/5 步加回读校验
     - 生成 .cmd 后按 ANSI 回读，确认占位符已替换且路径正确，否则抛错；
     - ghost_home.txt 写完后按 ANSI 回读，解出的路径必须等于仓库根。
       "文件内容对、编码错"这种静默故障以后会在安装阶段直接报错。

  4. 新增 output/fix-ghost-launcher.ps1（本次故障的现场修复工具）
     - 只重新生成安装目录的两个 .cmd 并重写 ghost_home.txt(GBK)，**不碰 dll/exe、
       不停不重启 WeaselServer**；带 -DryRun 诊断与 -Launch 当场验收。
     - 用法（管理员）：
       powershell -NoProfile -ExecutionPolicy Bypass -File "<repo>\output\fix-ghost-launcher.ps1" -Launch

生效条件
  - 装 v2.0.3 后需在【管理员 PowerShell】跑一次 install.ps1（或现场用
    output\fix-ghost-launcher.ps1），因为安装目录里的 .cmd 需要被重新生成。
  - 纯 git 更新不会改变已部署的 .cmd。

验收
  - 诊断：powershell -File "<repo>\output\fix-ghost-launcher.ps1" -DryRun
    应显示两个 .cmd 已内嵌正确路径、ghost_home.txt 按 ANSI 可正确读出；
  - 实测：-Launch 会直接以安装目录的 .cmd 起面板，并打印是否出现
    「LLM 数据面板 · cli_emojiless_RL」窗口。

不含：训练核心（损失 / 优化器 / 学习率 / 混料）与全部权重文件，与 v2.0.2 相同。
