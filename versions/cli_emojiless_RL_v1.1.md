cli_emojiless_RL_v1.1  (2026-09-09)

在 v1 基础上新增：离线记录模式、语言栏模式切换、兄弟节点并发批量前向。
基座：rime/weasel fork（去 emoji）；模型：Qwen3-0.6B-Base，主干 fp16 冻结 + lm_head fp32 可训。

架构
  WeaselServer.exe
    └─ (Job Object, KILL_ON_JOB_CLOSE) cmd.exe → ghost_service.cmd
         ├─ online  → pythonw online_server.py   候选服务 + 在线 RL
         └─ offline → pythonw offline_recorder.py 只记录上下文与提交段
  WeaselExpContextV0.exe（UIA 上下文钩子，winexe）

模式切换（三处入口，同一个模式文件）
  1. 语言栏右键菜单：LLM 在线推理 / LLM 离线记录（带单选勾）
  2. 托盘图标菜单：LLM 预测模式 → 在线推理 / 离线记录
  3. 命令行：python ghost_mode.py online|offline|status
  模式文件：%APPDATA%\Rime\ghost_mode.txt

在线模式
  /completion（llama-server 兼容 + pinyin 字段）给输入法出候选
  候选树：宽 20 / 深 10（显示与 RL 一致，RL 每层 beam 上限 20）
  采样：temperature 1.0 + repeat_penalty 1.3 / repeat_last_n 256
  拼音约束只作用于用户所在节点（depth 0），沿树逐字消费
  兄弟节点 8 路并发 + 服务端合并（一次 prefill + 一次批量前向）
  RL：reward = 叶子cum × 匹配比例；SGD lr 1e-4；梯度裁剪 1.0；空闲门控 0.3s
  保存：lm_head fp32，5 槽稀释（10s/5min/25min/2h/12h），按 updates 取最新

离线模式（无推理、无训练、零显存）
  记录 diag/segments.jsonl：ctx + 提交段（你/吃饭/了/吗），回退记 backspace
  训练：python offline_train.py（同一套奖励方法，共用五槽 checkpoint）

目录
  WeaselTSF/    TSF：候选注入、Tab 上屏、树维护、拼音路径、语言栏菜单
  WeaselServer/ Rime 后端 + 托管子进程 + 托盘菜单
  tools/        在线引擎、离线记录/训练、模式切换、看门狗、上下文钩子
  bin/          编译产物（weaselx64.dll / WeaselServer.exe / WeaselExpContextV0.exe）
  docs/         方案文档、项目报告、架构脱节审计
