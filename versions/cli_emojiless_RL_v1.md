cli_emojiless_RL_v1  (2026-09-09)

基座：rime/weasel fork（去 emoji，CLI 安装器）
模型：Qwen3-0.6B-Base，主干 fp16 冻结 + lm_head fp32 可训（155,582,464 可训参数）

架构
  WeaselServer.exe
    └─ (Job Object, KILL_ON_JOB_CLOSE) cmd.exe → ghost_service.cmd → pythonw online_server.py
         ├─ /completion  给输入法出候选（llama-server 兼容 + pinyin 字段）
         └─ RL 循环      上下文日志 → 奖励 → 单样本 SGD → 5 槽稀释落盘
  WeaselExpContextV0.exe（UIA 上下文钩子，随看门狗启动）

关键参数
  候选树：宽 20 / 深 10（显示侧与 RL 侧一致，RL 侧每层 beam 上限 20）
  采样：temperature 1.0 / top_k 0 / top_p 1.0 + repeat_penalty 1.3 / repeat_last_n 256
  上下文：光标前 256 字；拼音约束只作用于用户所在节点（depth 0）
  在线学习：SGD lr 1e-4，梯度范数裁剪 1.0，空闲门控 0.3s，单步超时 2s
  保存：lm_head fp32，5 槽（10s / 5min / 25min / 2h / 12h），启动按 updates 取最新
  并发：兄弟节点 8 路并发，服务端合并为一次 prefill + 一次批量前向

目录
  WeaselTSF/            输入法 TSF（候选注入、Tab 上屏、树维护、拼音路径、反馈上报）
  WeaselServer/         Rime 后端 + 预测服务子进程托管
  tools/LlamaTreeExp/   在线引擎服务与工具
  tools/WeaselExpContextV0/  上下文钩子（winexe，无控制台）
  bin/                  编译产物（weaselx64.dll / WeaselServer.exe）
  docs/                 方案文档、项目报告、架构脱节审计
