# weasel_LLM_RL

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.txt)

基于 [小狼毫 / Rime for Windows](https://github.com/rime/weasel) 改造的 LLM 输入法。
**候选树投机解码 + lm_head 在线强化学习**：模型在你打字的过程里持续更新，越用越准。

不是「接一个模型当接口」——主干冻结，只训练输出头，每一次上屏和回退都是训练信号。

---

## 重要提示

**用之前请先读完这一节。**

- **会采集输入信息**：为了训练和提升模型，本项目会记录你的输入内容（光标前的上下文、上屏的片段、回退删掉的内容）。这些数据**只落在本地** `tools/LlamaTreeExp/diag/` 目录，不会上传到任何服务器。
- **装上之后，用任何输入法都会被记录**：采集来自 UIA 上下文钩子，不依赖具体输入法。只要你还在用装了本项目的机器，哪怕切回小狼毫原生、微软拼音或搜狗，输入内容同样会被记进 `diag/` 目录。不想被记录就退出 WeaselServer 或删掉 `diag/` 下的数据。
- **不要在敏感场合使用**：正因为会采集输入，**不建议在任何对输入信息敏感的单位或场合使用**（涉密、金融、医疗、法务等）。
- **不支持英文联想**：目前只做中文预测，英文输入走小狼毫原生逻辑，没有 LLM 联想。
- **GPU 占用较大**：在线模式常驻约 1.5 - 2 GB 显存（0.6B 主干 fp16 + fp32 lm_head + 候选树缓存）。显存紧张时请切到离线模式。
- **冷启动效果一般**：当前基座（Qwen3-0.6B-Base）没有针对打字场景做过训练，刚开始用联想质量会比较一般，需要连续使用一段时间，让在线 RL 慢慢更新 lm_head，之后才会逐渐贴合你的输入习惯。
- **分享解码器请谨慎**：训练后的 lm_head 权重会带上你个人输入习惯的痕迹，公开分享前请想清楚。当然，也欢迎把你训练好的解码器提交回来。

---

## 它和「接个 LLM」有什么不同

| | 常见做法 | 本项目 |
|---|---|---|
| 模型 | 调用外部 API 或本地推理，参数不变 | 主干 fp16 冻结，lm_head fp32 在线更新 |
| 候选 | 一次生成、不中就丢 | 候选树边打边剪、边剪边长 |
| 反馈 | 无 | 上屏 = 正样本，回退 = 负样本 |
| 结果 | 你用得再多它也不会变 | 你在用它的同时训练它 |

---

## 架构

```
WeaselServer.exe
  └─ Job Object (KILL_ON_JOB_CLOSE)
       └─ cmd.exe -> ghost_service.cmd
            ├─ online  -> pythonw online_server.py     候选服务 + 在线 RL
            └─ offline -> pythonw offline_recorder.py  只记录上下文与提交段

WeaselExpContextV0.exe   UIA 上下文钩子（无窗口）
ghost_data_panel.py      数据面板：累计条数 / 待训练条数 / 训练入口
```

推理服务是 WeaselServer 的子进程，随输入法一起启动、一起退出，不需要手动开。

---

## 核心机制

### 候选树（投机解码式）

人打字不会等模型，所以模型在按键间隙里预先生成一棵树：

- **宽度 20**：每个节点取概率最高的 20 个续写
- **深度 10**：最多生长 10 层
- **标点截断**：遇到句号 / 逗号 / 问号 / 感叹号就停止该分支（省略号不截断）
- **剪枝再生长**：用户每敲一个字符，沿树保留匹配分支、丢掉兄弟分支，然后在新的叶子上继续生长

不是「生成一次、不中全丢」，而是「边打边剪、边剪边长」。

### 拼音约束

用户输入的拼音只作用在光标所在节点，沿候选路径逐字消费，保证候选同时满足拼音约束和语言模型概率。

### 在线强化学习

- **信号**：一次上屏 = 正样本，一次回退 = 负样本
- **奖励**：`reward = 叶子累计概率 × 字符匹配比例`（按 token 计，部分匹配给部分分）
- **更新**：只对 lm_head 做一步 SGD，`lr = 1e-4`，梯度裁剪 `1.0`
- **省算力**：生成候选树时已经做过前向，所以只需一次反向
- **并发**：兄弟节点 8 路并发，共享上下文缓存

### 检查点

5 个稀释槽位（10s / 5min / 25min / 2h / 12h），lm_head 以 fp32 落盘，启动时取 updates 最大的那个。

---

## 两种模式

| 模式 | 做什么 | 显存 |
|---|---|---|
| `online` | 候选树 + 在线 RL，边打边学 | 约 1.5 - 2 GB |
| `offline` | 只记录上下文和提交段，不推理不训练 | 0 |

**一条数据 = 一次提交断点**。「你吃饭了吗」分四次上屏，就是 `你` / `吃饭` / `了` / `吗` 四条；回退记一条 `backspace`，带上被删掉的文本。

离线模式下，数据面板可以一键「训练并丢弃」：用同一套奖励方法训练一遍，样本用完即删，不复用。

---

## 目录

```
WeaselTSF/             TSF：候选注入、Tab 上屏、树维护、拼音路径、语言栏菜单
WeaselServer/          Rime 后端 + 托管子进程 + 托盘菜单
tools/LlamaTreeExp/    在线引擎、离线记录/训练、数据面板、模式切换、看门狗
tools/WeaselExpContextV0/  UIA 上下文钩子
versions/              各版本归档 zip
```

---

## 快速开始

### 依赖

- Windows 10 / 11
- Visual Studio 2022（构建 TSF 与 WeaselServer）
- Python 3.10+，PyTorch（CUDA）、transformers、cryptography
- [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)

### 模型

放到 `<repo>/models/Qwen3-0.6B-Base`，或设置环境变量 `WEASEL_LLM_MODEL` 指向别处。

### 构建

```bat
git submodule update --init --recursive
build.bat
```

产物：`output/weaselx64.dll`、`output/WeaselServer.exe`。

### 部署

1. 把 `weaselx64.dll`、`WeaselServer.exe` 复制到 Rime 安装目录
2. 把 `tools/LlamaTreeExp/ghost_service.cmd`、`ghost_data_panel.cmd` 一并复制过去
3. 在安装目录建 `ghost_home.txt`，内容是本仓库根路径（一行）
4. 让注册表 `HKLM\SOFTWARE\Classes\CLSID\{A3F4CDED-B1E9-41EE-9CA6-7B4D0DE6CB0A}\InprocServer32` 指向新的 dll

### 运行

WeaselServer 启动时读取 `%APPDATA%\Rime\ghost_mode.txt`，按模式拉起对应服务。

切换模式有三种入口：

- 语言栏右键菜单：LLM 在线推理 / LLM 离线记录
- 托盘图标菜单
- 命令行：`python tools/LlamaTreeExp/ghost_mode.py online|offline|status`

---

## 配置

仓库里不含任何机器相关的绝对路径，全部通过环境变量或相对仓库根解析：

| 变量 | 作用 | 默认值 |
|---|---|---|
| `WEASEL_LLM_HOME` | 仓库根目录 | 脚本所在位置的上两级 |
| `WEASEL_LLM_MODEL` | HuggingFace 模型目录 | `<home>/models/Qwen3-0.6B-Base` |
| `WEASEL_LLM_GGUF` | llama.cpp 用的 GGUF 模型 | `<home>/models/Qwen3-0.6B-Base-Q8_0.gguf` |
| `WEASEL_LLM_SERVER` | `llama-server.exe` | `<home>/llama.cpp/llama-server.exe` |
| `WEASEL_LLM_DLL` | `llama.dll` | `<home>/llama.cpp/llama.dll` |
| `WEASEL_LLM_PYTHON` | `python.exe` | PATH |
| `WEASEL_LLM_PYTHONW` | `pythonw.exe` | PATH |
| `WEASEL_SERVER` | `WeaselServer.exe` | 注册表 CLSID 指向的目录 |
| `WEASEL_HOME` | Weasel 安装目录 | 注册表 CLSID 指向的目录 |

---

## 版本

| 版本 | 说明 |
|---|---|
| `cli_emojiless_RL_v1.2` | 数据面板、无头离线训练与进度、入口去 cmd 窗口、脚本编码修复 |
| `cli_emojiless_RL_v1.1` | 离线记录模式、语言栏模式切换、兄弟节点批量前向 |
| `cli_emojiless_RL_v1` | 在线 RL 引擎托管进 WeaselServer |

完整归档见 `versions/`。

---

## 许可与致谢

本项目是 [rime/weasel](https://github.com/rime/weasel) 的衍生作品，遵循 **GPLv3**（见 `LICENSE.txt`）。

### 代码主要贡献

- DeepSeek V4.1 Flash 0910
- DeepSeek V4 Flash 0731
- GLM-5.3-Flash

### 上游项目与依赖

- 中州韻輸入法引擎 / Rime Input Method Engine
- [librime](https://github.com/rime/librime)、[plum](https://github.com/rime/plum)
- Boost、curl、google-glog、Google Test、LevelDB、marisa-trie、OpenCC、WinSparkle、yaml-cpp、7-Zip

上游作者与贡献者：佛振、鄒旭、Xiangyan Sun、Prcuvu、nameoverflow、fxliang、Azuk 443 等，
完整名单见 [rime/weasel](https://github.com/rime/weasel)。

模型：[Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)（Apache-2.0）。
