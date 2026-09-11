# 版本归档约定

**规则**
- 每个可复现/可回滚的版本，统一打包成 versions/<版本名>.zip 提交到 git。
- 仓库内不再保留一坨坨实验快照目录；系统上只运行当前实验版本一个。
- 回滚：解压对应 versions/<版本名>.zip 到干净工作树，重新构建即可。

**当前 version 清单**
| zip | 含义 | 基座 |
|-----|------|------|
| cli_emojiless_exp_v0.3.zip | v0.3 外挂基线：原生输入法本体 + 外部上下文/候选树/RL 链路 | v0.3 |
| cli_emojiless_exp_v0.3_textshow.zip | TSF 内嵌预览实验（已废弃，记事本会把预览顶成真文本） | textshow |
| cli_emojiless_exp_v0.3_textshow_tsf_experiments.zip | 文本显示实验过程中 13 个 weaselx64.dll 快照 | textshow |
| WeaselOverlayV0_abandoned.zip | 外部窗口预览工具（已废弃，改用输入法候选框注入） | overlay |
| cli_emojiless_RL_v1.zip | 在线 RL 引擎托管进 WeaselServer | v1 |
| cli_emojiless_RL_v1.1.zip | 离线记录模式、语言栏模式切换、兄弟节点批量前向 | v1.1 |
| cli_emojiless_RL_v1.2.zip | 数据面板、无头离线训练与进度、入口去 cmd 窗口、脚本编码修复 | v1.2 |
| cli_emojiless_RL_v1.2.1.zip | 安装器从 Release 拉模型（-DownloadModels / -ModelsOnly / -Mirror） | v1.2.1 |
| cli_emojiless_RL_v1.3.zip | 拼音前缀匹配、top-k 免建树训练、checkpoint 同步防覆盖、Tab 防僵尸预测 | v1.3 |
| cli_emojiless_RL_v1.4.zip | 在线离线统一：逐 token 排名奖励、每次提交都训练、未命中走负样本 | v1.4 |
| cli_emojiless_RL_v2.0.zip | **第一个可正式训练的版本**：纯交叉熵 + 普通 SGD + lr 1e-5 + 打字/小说 25% 混料；新增 train_mix.py 与三指标验收；清理早期原型与死代码 | v2.0 |

**当前工作树**
- 基座：cli_emojiless_RL_v2.0（HEAD）
- 活动实验：tools\LlamaTreeExp + tools\WeaselExpContextV0（外部钩子/候选树/训练）
- 定型配置：纯交叉熵 / 普通 SGD / lr 1e-5 / 梯度裁剪关闭 / 打字:小说 = 25:75
- 下一步：验证超大数据量下的泛化（当前只到 1 万字、17001 步；同域 +3 点，换书换文体无数据）
