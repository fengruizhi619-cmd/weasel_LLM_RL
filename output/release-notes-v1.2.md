自强化 LLM 输入法（小狼毫 + 冻结主干 + 在线 RL 解码器）。

## 资产

| 资产 | 大小 | 放到哪 |
| --- | --- | --- |
| `Qwen3-0.6B-Base.zip.001` / `.002` / `.003` | 1.2 GB（分 3 段） | 合并解压到 `<repo>/models/Qwen3-0.6B-Base` |
| `lm_head_t0.pt` | 622 MB | `<repo>/tools/LlamaTreeExp/diag/checkpoints_online/` |

主干是 fp16、解码器是 fp32，这两块是拼合法的组成部分，别换成官方原模型或别处来的权重。

## 为什么直接把模型传上来

- 国内访问 GitHub / HuggingFace 都不稳定，自动拉取容易下不下来
- 本项目用的是拼合法：主干 fp16 + 解码器 fp32，精度不同，直接拉官方原模型会出现各种不匹配问题
- 仓库大不额外付费，直接给省一步

## 部署

```powershell
.\install.ps1 -ModelsOnly      # 只下模型：自动下载、合并分段、解压
.\install.ps1                  # 装输入法本体
```

或者一条命令全干：

```powershell
.\install.ps1 -DownloadModels
```

国内直连慢的话加镜像：

```powershell
.\install.ps1 -ModelsOnly -Mirror https://gh.xxooo.cf/
```

手动部署：下载三个分段 -> 二进制合并成 `Qwen3-0.6B-Base.zip` -> 解压到 `<repo>/models/Qwen3-0.6B-Base`，
再把 `lm_head_t0.pt` 放到 `<repo>/tools/LlamaTreeExp/diag/checkpoints_online/`。

## 注意

- 目前输入法不支持英文联想
- GPU 占用较大
- 项目不建议在任何对输入信息敏感的单位或场合使用；为了提升和训练模型会采集输入信息用于训练，训练数据存在本地
- 目前使用的模型未针对性训练，联想效果不是很好，需要使用一段时间后才能更加贴合使用习惯
- 请谨慎分享解码器部分，也欢迎提供训练好的解码器
