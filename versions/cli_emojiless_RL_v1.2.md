cli_emojiless_RL_v1.2  (2026-09-09)

在 v1.1 基础上新增：独立数据记录器与数据面板、离线训练无头化与实时进度、面板入口无窗口。

基座：rime/weasel fork（去 emoji）；模型：Qwen3-0.6B-Base，主干 fp16 冻结 + lm_head fp32 可训。

v1.1 -> v1.2 增量
  1. 数据记录器与输入法解耦：offline_recorder.py 后台常驻，读 UIA 钩子日志，
     不依赖具体输入法；watchdog 保活，任何输入法都能收集样本
  2. 数据面板 ghost_data_panel.py / .cmd：累计条数、待训练条数、训练入口
     「训练并丢弃」——样本一次用完即丢，训练成功后清空
  3. 离线训练无头化：面板用 CREATE_NO_WINDOW 启动 python.exe，stdout 走管道
     读进度，全程无控制台窗口
  4. 训练不阻塞 UI：stdout 由后台线程读进 queue，UI 线程 after(200) 轮询；
     实测 UI 事件最大延迟 0.196s；进度条按 [progress] i/total 更新
  5. 计数口径：待训练 = segments.jsonl 行数 - segments.seen；
     训练后只删除已消费的行数，训练期间新写入的记录保留
  6. 面板入口无窗口：WeaselServerApp.cpp 用 SW_HIDE 打开 ghost_data_panel.cmd
  7. 修复 .cmd 编码：ghost_data_panel.cmd / ghost_service.cmd 由 UTF-8 改 GBK+CRLF
     （cmd.exe 按 ANSI 解析批处理，UTF-8 会让中文路径乱码，pythonw 静默失败）

一条数据 = 一次提交断点
  记录器比较前后两次上下文的最长公共前缀：
    变长 -> 一条 commit（上屏段），变短 -> 一条 backspace（回退段）
  「你吃饭了吗」分四次上屏 = 你 / 吃饭 / 了 / 吗 共 4 条；一次回退记 1 条

架构（同 v1.1）
  WeaselServer.exe
    └─ (Job Object, KILL_ON_JOB_CLOSE) cmd.exe -> ghost_service.cmd
         ├─ online  -> pythonw online_server.py    候选服务 + 在线 RL
         └─ offline -> pythonw offline_recorder.py 只记录上下文与提交段
  WeaselExpContextV0.exe（UIA 上下文钩子，winexe）
  数据面板：WeaselServer 托盘/语言栏菜单 -> ghost_data_panel.cmd（无窗口）

在线模式
  /completion（llama-server 兼容 + pinyin 字段）
  候选树宽 20 / 深 10；temperature 1.0 + repeat_penalty 1.3 / repeat_last_n 256
  拼音约束只作用于用户所在节点，沿树逐字消费
  兄弟节点 8 路并发 + 服务端批量合并（一次 prefill + 一次批量前向）
  RL：reward = 叶子cum × 匹配比例；SGD lr 1e-4；梯度裁剪 1.0；空闲门控 0.3s
  保存：lm_head fp32，5 槽稀释（10s/5min/25min/2h/12h），按 updates 取最新

离线模式（无推理、无训练、零显存）
  记录 diag/segments.jsonl：ctx + 提交段（你/吃饭/了/吗），回退记 backspace
  训练：python offline_train.py（同一套奖励方法，共用五槽 checkpoint）
  面板：训练并丢弃（无窗口 + 进度条）

目录
  WeaselTSF/    TSF：候选注入、Tab 上屏、树维护、拼音路径、语言栏菜单
  WeaselServer/ Rime 后端 + 托管子进程 + 托盘菜单
  tools/        在线引擎、离线记录/训练、数据面板、模式切换、看门狗、上下文钩子
  bin/          编译产物（weaselx64.dll / WeaselServer.exe / WeaselExpContextV0.exe）
  docs/         方案文档、项目报告、架构脱节审计

v1.2 实测
  离线训练 58 条真实数据：hits=30 steps=34，updates 397 -> 465，数据按设计丢弃
  面板入口：ShellExecuteW(SW_HIDE) 打开 ghost_data_panel.cmd，面板进程正常且无 cmd 窗口
  托管链路：重启 WeaselServer 后 ghost_service.cmd 正确 cd 到工作区并拉起记录器
