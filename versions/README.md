# 版本归档约定

**规则**
- 每个可复现/可回滚的版本，统一打包成 ersions/<版本名>.zip 提交到 git。
- 仓库内不再保留一坨坨实验快照目录；系统上只运行当前实验版本一个。
- 回滚：解压对应 ersions/<版本名>.zip 到干净工作树，重新构建即可。

**当前 version 清单**
| zip | 含义 | 基座 |
|-----|------|------|
| cli_emojiless_exp_v0.3.zip | v0.3 外挂基线：原生输入法本体 + 外部上下文/候选树/RL 链路 | v0.3 |
| cli_emojiless_exp_v0.3_textshow.zip | TSF 内嵌预览实验（已废弃，记事本会把预览顶成真文本） | textshow |
| cli_emojiless_exp_v0.3_textshow_tsf_experiments.zip | 文本显示实验过程中 13 个 weaselx64.dll 快照 | textshow |
| WeaselOverlayV0_abandoned.zip | 外部窗口预览工具（已废弃，改用输入法候选框注入） | overlay |

**当前工作树**
- 基座：cli_emojiless_exp_v0.3（外挂基线，HEAD 496ad2f）
- 活动实验：	ools\LlamaTreeExp + 	ools\WeaselExpContextV0（外部钩子/候选树/RL）
- 下一步实验基座：v0.3，方向＝把 LLM 预测注入输入法候选框（候选标 Tab，面板保持 8s）
