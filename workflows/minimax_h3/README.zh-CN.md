# 当前角色的 H3 双视角视频

使用本机已安装的 ComfyUI 原生 H3 节点与已下载模型，生成“下蹲蓄力、小步前探格挡、转髋直拳、提膝平衡、落脚回位”的八秒连续动作。

## 文件与使用

- `MedievalGirl_ComplexAction.workflow.json`：在 ComfyUI 中打开的工作流。
- `MedievalGirl_ComplexAction.api.json`：同参数的 API 工作流。
- `MedievalGirl_ComplexAction.prompt.txt`：完整中文提示词，已写入两份工作流。修改提示词时同时更新工作流中的 `MiniMaxH3ImageToVideo` 节点。

本机已将工作流放入 ComfyUI 的 `user/default/workflows/Video2Rig/MiniMaxH3_MedievalGirl.json`。输入图来自本仓库 `work/MedievalGirlTraining/start/start.png`，复制为 ComfyUI 的 `input/video2rig_MedievalGirlTraining_start.png`。角色图片保留在本地，不随 Git 分发；在另一台机器使用时须自行提供该输入。

打开工作流，确认输入图片，点击运行。视频保存到 ComfyUI 输出目录下的 `video2rig/MedievalGirl/H3_ComplexAction_*.mp4`。本机输出目录是 `F:\AI_Pipeline\Outputs`。

本次使用现有环境启动独立的本地服务，浏览器地址为 `http://127.0.0.1:8188`。服务关闭后，可在没有其它进程占用该端口时用以下命令重开：

```powershell
Set-Location F:\AI_Pipeline\Tools\ComfyUI
& F:\AI_Pipeline\Envs\comfyui\python.exe -X utf8 -u main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-all-custom-nodes --disable-partner-nodes --reserve-vram 2 --cache-none --output-directory F:\AI_Pipeline\Outputs
```

这些启动参数只作用于本次服务，不修改已有启动器；工作流本身只依赖原生节点。

## 已配置参数

| 项目 | 设置 |
| --- | --- |
| 主模型 | `minimax_h3_fl2va_pruned_w6a8.safetensors` |
| 文本与图像编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`，类型 `minimax` |
| 视频 VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` |
| 输出 | 1024×576，24 fps，192 帧，即 8 秒 |
| 采样 | `res_multistep` + `simple`，20 步，denoise 1，BasicGuider（CFG 1） |
| 种子 | 2026102701，固定 |
| 首尾帧 | 同一张双视角标定渲染图 |
| 视频解码 | 原生分块解码，512 像素空间块、64 帧时间块 |
| 保存 | MP4 / H.264，CRF 18 |

1024×576 与原始 1920×1080 首帧严格同为 16:9，输出的宽、高均为 32 的倍数。原生首帧节点会缩放首图；这里不改变宽高比、不移动中线，也不分别裁剪两栏。8 秒对应 H3 的合法帧数 `17×11+5=192`。

## 视角与关节对应

骨架、首帧和相机均来自当前训练工作包。左栏是从世界坐标 -Y 方向观察的正面机位；右栏相机位于 +X，观察的是角色的**解剖学左侧**，起始人物朝画面左方。“右栏”不是“人物右侧”。两相机等高、等比例、相差九十度；首尾输入天然属于同一副骨架。

提示词明确两栏共享同一动作时钟，角色自身左右身份不变，各动作极值及落脚同时发生，对应关节保持相同竖直高度。整段只有一个固定机位镜头，并在结尾回到起始姿势。

首尾帧锚定和文字约束不能把 H3 的中间帧变成严格三维投影。若需要逐帧严格同骨架，应使用同一三维动作的双视角渲染或逐帧控制；当前流程的中间帧同步程度需要看生成结果。

本机试跑已完成：RTX 3090，1024×576、192 帧、24 fps，端到端约 14 分 10 秒。抽查 0～8 秒关键画面，固定分栏与机位保持，出拳、提膝、回位基本同步；中间帧仍有少量关节位置偏差。视频文件为 `H3_ComplexAction_00001_.mp4`，本地查看图位于 `work/MedievalGirlTraining/Review/H3_ComplexAction.png`。

## 官方依据

- [MiniMax 官方基础模式提示词指南](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)：首尾帧对应说明、单镜头连续动作、三个提示词字段。
- [ComfyUI H3 提示词指南](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-prompt-guide)：正向表达、BasicGuider 不使用负面提示词分支。
- [ComfyUI H3 使用指南](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)：W6A8、帧数规则及基础采样设置。

官方文字示例推荐英文；本项目按中文提示词要求撰写，保留 `Picture`、`Shot` 和三个字段名。没有另设负面提示词，也没有改动 ComfyUI 或 video2rig 核心代码。
