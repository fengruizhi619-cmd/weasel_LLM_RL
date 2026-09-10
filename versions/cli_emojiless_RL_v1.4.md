cli_emojiless_RL_v1.4  (2026-09-10)

在 v1.3 基础上：在线与离线训练统一为同一套奖励规则——逐 token 排名奖励、逐层相加。

基座：rime/weasel fork（去 emoji）；模型：Qwen3-0.6B-Base，主干 fp16 冻结 + lm_head fp32 可训。

v1.3 -> v1.4 增量

  1. 奖励规则（两条链统一）
     - 每一步取当前可达候选的 top-k，按累计概率排名，真值 token 的奖励由排名决定
       harmonic: 1, 1/2, 1/3, 1/4, 1/5 ...（另有 linear / exp 可选）
     - 每个 token 逐层各自计算，序列总奖励 = 各层相加。
       即 a->b(rank2)->c(rank1)->d(rank5) 的总奖励为 g(2)+g(1)+g(5)
     - 长度因此成为加分项。旧公式为 cum x ratio，cum 是逐 token 概率连乘：命中
       10 字约 0.3^10 = 6e-6，而命中 1 字有 0.3 量级，差五个数量级，长命中在
       梯度里等于不存在
  2. 在线链路与离线链路对齐
     - 在线 accept 从「cum x ratio + 只强化 target_ids[0]」改为走同一套逐字路径
       （train_sequence：KV 缓存逐字前进，每个 token 各自一步）
     - 在线不再只挑命中样本：每次提交都作为样本入队；未命中仍产出负样本步
       （对错误的 top-1 做 unlikelihood），与离线一致
     - 常量共用：--topk 20 / --rank-reward harmonic / --miss-weight 0.3 / --grad-clip 1.0
  3. 离线的差别只在算力
     - 一次过完积压记录；--include-ctx 可把记录里的上下文也当监督
     - 数学与在线完全一致
  4. 可观测性
     - 在线日志：[RL] accept text=... steps=N hits=M reward=X
     - 离线日志：累计奖励与每命中平均奖励

架构（同 v1.3）
  WeaselServer.exe
    └─ (Job Object, KILL_ON_JOB_CLOSE) cmd.exe -> ghost_service.cmd
         ├─ online  -> pythonw online_server.py     候选服务 + 在线 RL
         └─ offline -> pythonw offline_recorder.py  只记录上下文与提交段
  WeaselExpContextV0.exe（UIA 上下文钩子，winexe）
  数据面板：WeaselServer 托盘/语言栏菜单 -> ghost_data_panel.cmd（无窗口）

在线模式
  候选树宽 20 / 深 10；temperature 1.0 + repeat_penalty 1.3 / repeat_last_n 256
  拼音约束只作用于用户所在节点，沿树逐字消费（前缀匹配）
  训练：逐 token 排名奖励；SGD lr 1e-4；梯度裁剪 1.0；空闲门控 0.3s；步超时 2.0s
  保存：lm_head fp32，5 槽稀释（10s/5min/25min/2h/12h），按 updates 取最新；
        回写前先采纳磁盘上更新的权重（v1.3 起）

v1.4 实测
  奖励映射取值：
    harmonic  rank1-5 = 1.0 / 0.5 / 0.333 / 0.25 / 0.2
    linear    rank1-5 = 1.0 / 0.95 / 0.90 / 0.85 / 0.80
    exp       rank1-5 = 1.0 / 0.5 / 0.25 / 0.125 / 0.0625
  离线 dry-run（--mode fast，412 条）：steps=357 hits=183（51.3%）reward=113.6
    平均 0.621/命中，37 秒；旧公式下同批命中只有 0.05~0.2 量级，且被概率绝对值绑架
  在线：服务重启后 resumed updates=3291；新格式日志 [RL] accept ... steps/hits/reward
