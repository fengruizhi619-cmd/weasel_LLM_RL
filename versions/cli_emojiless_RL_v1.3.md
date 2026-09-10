cli_emojiless_RL_v1.3  (2026-09-10)

在 v1.2.1 基础上新增：拼音约束改为前缀匹配并全程生效、训练改为 top-k 免建树、
离线训练成果不再被在线服务覆盖、Tab 只提交仍然有效的预测。

基座：rime/weasel fork（去 emoji）；模型：Qwen3-0.6B-Base，主干 fp16 冻结 + lm_head fp32 可训。

v1.2.1 -> v1.3 增量

  1. 拼音约束（服务端两处）
     - token_pinyin() 改取首字音节：原来返回整个 token 的拼音（「大概」-> dagai），
       输入 daga 时第一个字就把整段输入消费完，后续路径完全失去约束——这就是
       「只匹配第一个字的拼音」的根因
     - 过滤改双向前缀：tp.startswith(pinyin) 或 pinyin.startswith(tp)，
       否则输入 daga 时「大」(da) 会被判为不匹配整批踢掉
     - 有拼音约束时不参与动态收窄（实测负载下候选会被从 200 压到 55，找不到匹配项）

  2. 候选树（客户端两处）
     - BestSuffix 在拼音没消费完又找不到匹配分支时停住，不再 remaining.clear() 后
       退回无约束挑最高概率分支（那正是预览里乱蹦无关字的来源）
     - SetPreedit 在拼音由空变非空时打 seed 标记，worker 用拼音过滤重建根层，
       保证打字期间根层真的有符合拼音的分支

  3. 训练速度与数据利用率
     - 新增 TreeEngine.train_sequence()：一次冻结主干 prefill + 每字一次单 token
       decode（主干冻结，KV 缓存跨头更新依然有效），每步取 top-k 与用户实打字比对，
       奖励 = 概率 × (1 - 排名/k)；未命中不丢样本，对错误的 top-1 做 unlikelihood
     - find_best_reward 修死代码：typed.startswith(path) 与 len(typed)<len(path)
       互斥，部分命中分支永不可达，改为双向匹配给 len(typed)/len(path) 的部分分
     - offline_train 新增 --mode fast(默认)/tree、--topk、--miss-weight、--include-ctx
     - dry-run 不再写 segments.seen（原来会「没训练却把数据标记成已消费」）

  4. 权重一致性（在线与离线共享同一份头）
     - CheckpointManager.sync_from_disk()：每次 save_epoch 前扫 5 个槽位 mtime，
       发现比上次写入更新且 updates 更大的槽位就先加载、再决定是否回写。
       否则在线服务会按稀释槽位把内存里的旧头写回去，静默覆盖刚训练出来的结果
     - online_server.save_epoch() 采纳新头后 reset_cache()

  5. 语料加密密钥
     - 密钥钉到 diag/corpus.key：首次运行用现有 USERNAME 派生方式写入（保证已写数据
       仍可读），之后只认这个文件
     - 原因：原来靠 WEASEL_CORPUS_KEY 或用户名派生，换运行环境就派生第二把钥匙，
       corpus.jsonl 最前面 14 条因此永久读不出（AES-GCM InvalidTag，base64 与长度正常）

  6. Tab 只提交有效预测
     - GhostEngine::OnDocumentPrefix()：与预测生成时的 visible_snapshot_.prefix 不一致
       就丢弃预览（只清可见文本，不重建树，开销极小）
     - WeaselTSF::_ReadGhostPrefix() / _SyncGhostDocument()：每次编辑上报当前 caret 前缀
     - TextEditSink::OnEndEdit 先读 pEditRecord 判断是否真有文本改动/选区变化
       （原来枚举了却忽略结果），有变化才同步
     - _TryCommitPrediction 增加一致性校验：要求 engine->VisiblePrediction() 与候选
       列表里持有的文本相同（候选列表 500ms 才刷新，可能握着引擎已丢弃的预测）
     - 根因：_expSnapshotPending 全项目只在 IME 上屏时置真，普通打字/退格/空格都不会
       让 TSF 重读文档；而 GhostEngine::HandleKey 是死代码从未被调用，于是僵尸预测被
       Tab 补上屏

架构（同 v1.2.1）
  WeaselServer.exe
    └─ (Job Object, KILL_ON_JOB_CLOSE) cmd.exe -> ghost_service.cmd
         ├─ online  -> pythonw online_server.py     候选服务 + 在线 RL
         └─ offline -> pythonw offline_recorder.py  只记录上下文与提交段
  WeaselExpContextV0.exe（UIA 上下文钩子，winexe）
  数据面板：WeaselServer 托盘/语言栏菜单 -> ghost_data_panel.cmd（无窗口）

在线模式
  候选树宽 20 / 深 10；temperature 1.0 + repeat_penalty 1.3 / repeat_last_n 256
  拼音约束只作用于用户所在节点，沿树逐字消费（前缀匹配）
  RL：reward = 叶子cum × 匹配比例；SGD lr 1e-4；梯度裁剪 1.0
  保存：lm_head fp32，5 槽稀释（10s/5min/25min/2h/12h），按 updates 取最新

v1.3 实测
  拼音：prompt=我觉得，pinyin = daga / dagai / dak 均返回 2 条（大家、大，首字拼音 da）；
        改前这三者拿到的都是未过滤的 200 条
  训练：582 条 -> 9934 步 / 命中 5759（58.0%）/ 627 秒 / updates 2536 -> 3118
        效率约 16-27 步/秒，比建树模式快一个数量级以上
  权重：模拟「内存里只有 100 步的旧服务」，save_epoch 采纳磁盘上 3119 步的头，
        且磁盘未被 100 覆盖
  语料：换新代码后 corpus.jsonl 668 条、segments.jsonl 620 条仍可解密
  Tab：部署后僵尸预测场景（仅退格/空格后按 Tab）不再提交
