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

**当前工作树**
- 基座：cli_emojiless_RL_v1.3（HEAD）
- 活动实验：tools\LlamaTreeExp + tools\WeaselExpContextV0（外部钩子/候选树/RL）
- 下一步：训练已提速（top-k 免建树，约 16-27 步/秒），继续长期使用积累语料；在线与离线共享同一份头，保存前会先采纳磁盘上更新的权重
