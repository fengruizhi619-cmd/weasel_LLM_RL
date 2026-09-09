cli_emojiless_RL_v1.2.1  (2026-09-09)

在 v1.2 基础上新增：安装器从 GitHub Release 拉取模型，一键部署闭环。

基座：rime/weasel fork（去 emoji）；模型：Qwen3-0.6B-Base，主干 fp16 冻结 + lm_head fp32 可训。

v1.2 -> v1.2.1 增量
  1. install.ps1 新增三个开关
     - -ModelsOnly     只下模型，不装输入法（不需要管理员）
     - -DownloadModels 装完输入法后顺带拉模型
     - -Mirror <前缀>  国内网络走镜像，如 https://gh.xxooo.cf/
     环境变量 WEASEL_LLM_MIRROR 可作为镜像默认值
  2. 模型来源改为 Release 资产
     - Qwen3-0.6B-Base.zip.001/.002/.003（1.2GB，分 3 段）
       -> 二进制合并成 Qwen3-0.6B-Base.zip -> 解压到 models/Qwen3-0.6B-Base
     - lm_head_t0.pt（622MB）-> tools/LlamaTreeExp/diag/checkpoints_online/
     - 目标已存在则跳过，重复执行安全
  3. README 模型章节改指向 Release 资产，并写明主干 fp16 + 解码器 fp32 属拼合法，
     不能换成官方原模型或别处来的权重
  4. 安装完成后按实际状态打印主干 / 解码器是否就位

为什么分 3 段
  单次 1.2GB 上传在 762MB 处被连接重置（GitHub 不支持断点续传），切 400MB 后稳定。

Release
  https://github.com/fengruizhi619-cmd/weasel_LLM_RL/releases/tag/cli_emojiless_RL_v1.2.1
  4 个资产的 SHA256 与 GitHub digest 全部一致。

v1.2.1 实测
  - install.ps1 语法解析通过；-ModelsOnly -DryRun 与真实跳过分支均正常
  - 三段合并后哈希 5cff1bdd 与原 zip 一致
  - 镜像 gh.xxooo.cf range GET 返回 206，取回首 1MB 与本地逐字节一致
  - 直连同一请求只回 699KB 即断，故国内建议 -Mirror

架构（同 v1.2）
  WeaselServer.exe
    └─ (Job Object, KILL_ON_JOB_CLOSE) cmd.exe -> ghost_service.cmd
         ├─ online  -> pythonw online_server.py     候选服务 + 在线 RL
         └─ offline -> pythonw offline_recorder.py  只记录上下文与提交段
  WeaselExpContextV0.exe（UIA 上下文钩子，winexe）
  数据面板：WeaselServer 托盘/语言栏菜单 -> ghost_data_panel.cmd（无窗口）

在线模式
  候选树宽 20 / 深 10；temperature 1.0 + repeat_penalty 1.3 / repeat_last_n 256
  拼音约束只作用于用户所在节点，沿树逐字消费
  RL：reward = 叶子cum × 匹配比例；SGD lr 1e-4；梯度裁剪 1.0
  保存：lm_head fp32，5 槽稀释（10s/5min/25min/2h/12h），按 updates 取最新
