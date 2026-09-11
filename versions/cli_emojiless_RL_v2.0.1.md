cli_emojiless_RL_v2.0.1  (2026-09-11)

在 v2.0 基础上的小版本：只动工具与界面，训练核心与权重都没碰。

v2.0 -> v2.0.1 增量

  1. 解码器库与切换（新增 tools/LlamaTreeExp/head_lib.py）
     - diag/heads/<name>.pt + <name>.json：把训练出来的 lm_head 收进一个库，
       每条带备注、updates、时间戳
     - list / import / snapshot / activate 四个动作，命令行与面板都能用
     - 切换 = 写 checkpoints_online/lm_head_t0.pt，并把 updates 提成"该目录所有
       槽的最大值 +1"。在线服务每 15 秒 sync_from_disk() 一次，只有磁盘上的
       updates 更大才采纳——不提这一格，换过去也不生效，非得重启服务
     - 切换前自动把当前那份收进库（按 updates+timestamp 判重），随时能换回来
     - 只写 t0，其余四个稀释槽不动，回滚阶梯不丢
  2. 数据面板新增「解码器（lm_head）」区
     - 下拉框（名字 / updates / 备注）+ 切换 / 当前入库 / 刷新
     - 下面一行显示当前线上的 updates 与时间戳
  3. 修面板被裁切的 bug
     - 根因：面板是固定尺寸 geometry("480x470") + resizable(False, False)，
       而新增的解码器行把布局的自然宽度顶到 668px，右边约 188px 被裁掉
     - 修法：窗口改 820x640 + minsize(820,600) + 允许缩放；下拉框 58 -> 46 宽、
       显示文本缩短；日志区加 sticky="nsew" 与行列权重，随窗口拉伸
     - 验收：脚本遍历所有控件、比较右边界与窗口宽度，超出 0 个
  4. 线上解码器已切换到 mix_base（当天训练的最新一份）

不含：训练核心（损失 / 优化器 / 学习率 / 混料）与权重文件，与 v2.0 相同。
