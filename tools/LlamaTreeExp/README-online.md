# 在线统一引擎（S1-S6）

一个 PyTorch 模型同时干两件事：给输入法出候选、吃用户的采纳/回退做在线 SGD。
主干 fp16 冻结，lm_head fp32 可训；训练与推理共用同一份权重，梯度立刻影响下一棵树。

## 一键启动

```
start-online.cmd     启动 WeaselServer + 上下文钩子 + 引擎服务(8081)
stop-online.cmd      停引擎和钩子，WeaselServer 保持运行
```

## 接口

| 端点 | 说明 |
|---|---|
| POST /completion | llama-server 兼容：prompt/n_probs/temperature/top_k/top_p/min_p/repeat_penalty/repeat_last_n -> completion_probabilities[0].top_probs[{token,prob,id,pinyin}] |
| GET /health | 存活检查 |
| GET /metrics | 请求数/缓存命中/延迟分位/RL 次数/回退次数/回滚次数 |
| POST /feedback | 预留：{"kind":"accept"|"reject"} |

额外字段 `pinyin`（可选）：只返回拼音前缀匹配的候选（S3）。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| --dtype | float16 | 主干精度（头恒为 fp32） |
| -n/--width | 20 | 候选树宽度 |
| -d/--depth | 2 | 候选树深度 |
| --rl-lr | 1e-4 | 单样本 SGD 学习率 |
| --idle-gate | 0.3 | 距上次请求不足该秒数则不训练（S5） |
| --grad-clip | 1.0 | 梯度范数裁剪（S5） |
| --step-timeout | 2.0 | 单步超时则回滚该步（S5） |
| --eval-every | 50 | 每 N 次更新做一次回归探针（S5） |
| --rollback-margin | 0.05 | 探针低于基线该幅度则回滚到 t2h 槽（S5） |

## 数据

- checkpoint：diag/checkpoints_online/lm_head_t0|t5m|t25m|t2h|t12h.pt（只存头，5 槽时间稀释，启动 resume）
- 语料：diag/corpus.jsonl（XOR+base64 混淆落盘，键由 WEASEL_CORPUS_KEY/用户名派生）
- 日志：diag/online-server.log、diag/exp-run-v02.log

## 兼容性 / 降级

- 引擎没起来：输入法照常工作，只是没有 ghost 预览（TSF 侧 HTTP 失败即返回空）。
- 临时关掉联想：给目标进程设 WEASEL_GHOST_DISABLE=1。
- 换推理地址：设 WEASEL_GHOST_URL（默认 http://127.0.0.1:8081/completion）。
- 应用黑名单：目前靠上述环境变量 + 逐应用验证，矩阵见 design/v0.3/weasel-llm-v0.3-build-plan.md。

## P3 增强

- 回退分类：predicted-reject（w=1.0）/ typing-reject（w=0.3）/ replace（w=0.5），见 `/metrics` 的 `backspace_*`。
- 命中率联动：`--hit-rate-high 0.8` / `--hit-rate-low 0.4`，命中率高时收窄候选省算力。
- 触发时机：引擎轮询 `/metrics` 的 accept_rate 自适应空闲阈值（400/800/1200ms）；上下文 < 2 字不预测。
- 延迟结论：FP8 更慢、fp16 头无收益、Windows 无 Triton；有效的是 logits LRU + token 缓存。
- `--prefetch` 默认关闭：预取会占模型锁，墙钟时间不变。

## 离线记录模式（无推理、无训练）

```
python ghost_mode.py status     # 看当前模式与进程
python ghost_mode.py offline    # 切到离线记录：停引擎、起记录器
python ghost_mode.py online     # 切回在线推理：停记录器、起引擎
```

模式文件：`%APPDATA%\Rime\ghost_mode.txt`（内容 `online` / `offline`）。
WeaselServer 启动时由 `ghost_service.cmd` 读它决定托管哪个进程；IME 侧（v18+）
读到 offline 就完全不再请求预测。

离线记录内容：`diag/segments.jsonl`，每条是「上下文 + 一个提交断点」，
例如「你吃饭了吗」会记成

```
ctx=""        segment="你"
ctx="你"      segment="吃饭"
ctx="你吃饭"  segment="了"
ctx="你吃饭了" segment="吗"
```

回退另记 `kind="backspace"`（被删掉的文本），替换记 `kind="replace"`。
同一个加密格式（AES-256-GCM），可用 `decrypt_corpus.py` 审计。

离线训练：

```
python offline_train.py                 # 训练 segments.jsonl 里的新记录
python offline_train.py --dry-run       # 只看命中率/奖励，不更新
python offline_train.py --epochs 3
```

训练方法与在线完全一致：从 ctx 建候选树 → `reward = 叶子cum × 字符匹配比例`
→ 对 lm_head 做一步 SGD；回退用 unlikelihood。checkpoint 共用同一套五槽，
在线/离线是同一条权重血脉。

## 后台数据采集（与输入法无关）

记录器现在**不依赖小狼毫**：无论你用微软拼音、搜狗还是小狼毫，只要钩子
（`WeaselExpContextV0.exe`，UIA 订阅文本变化）在跑，`offline_recorder.py`
就会在后台持续把「上下文 + 提交段」写进 `diag/segments.jsonl`。

- 常驻方式：看门狗计划任务（登录时 + 每 5 分钟）确保钩子和记录器都在；
  记录器有单实例锁（`diag/recorder.lock`），多个启动方不会重复。
- 记录器独立于 WeaselServer：切换模式或重启小狼毫都不会打断采集。
- 在线模式下记录器同样在跑，所以在线/离线的数据都在积累。

`ghost_mode.py status` 会同时显示模式、引擎、记录器和已记录的段数。
