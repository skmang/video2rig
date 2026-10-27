# H3 八秒工作流历史对照

后续动作改用 [关键姿势工具](../../tools/keyframe_motion/README.zh-CN.md) 准备首尾图和动作包，遵循 [生产规范](../../docs/MOTION-KEYFRAMES.zh-CN.md)：短动作 1～3 秒，长动作 3～5 秒。本页保留已跑过的八秒配置与结果作对照；该图同时作为新准备工具的原生节点模板，片长、提示词与首尾输入由动作合同覆盖。

使用本机已安装的 ComfyUI 原生 H3 节点与已下载模型，配置“下蹲蓄力、小步前探格挡、转髋直拳、提膝平衡、落脚站稳”的八秒连续动作，优先保持关节、轮廓和接触关系清楚。当前提示词采用自然连贯的战斗组合描述，不按秒规定关节运动。

## 文件与使用

- `MedievalGirl_ComplexAction.workflow.json`：在 ComfyUI 中打开的工作流。
- `MedievalGirl_ComplexAction.api.json`：同参数的 API 工作流。
- `MedievalGirl_ComplexAction.prompt.txt`：完整中文提示词，已写入两份工作流。修改提示词时同时更新工作流中的 `MiniMaxH3ImageToVideo` 节点。

本机已将工作流放入 ComfyUI 的 `user/default/workflows/Video2Rig/MiniMaxH3_MedievalGirl.json`。输入图来自本仓库 `work/MedievalGirlTraining/start/start.png`，复制为 ComfyUI 的 `input/video2rig_MedievalGirlTraining_start.png`。角色图片保留在本地，不随 Git 分发；在另一台机器使用时须自行提供该输入。

打开工作流，确认输入图片，点击运行。视频保存到 ComfyUI 输出目录下的 `video2rig/MedievalGirl/H3_NaturalAction_*.mp4`。本机输出目录是 `F:\AI_Pipeline\Outputs`。

本次使用现有环境启动独立的本地服务，浏览器地址为 `http://127.0.0.1:8188`。服务关闭后，可在没有其它进程占用该端口时用以下命令重开：

```powershell
Set-Location F:\AI_Pipeline\Tools\ComfyUI
& F:\AI_Pipeline\Envs\comfyui\python.exe -X utf8 -u main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-all-custom-nodes --disable-partner-nodes --reserve-vram 2 --cache-ram 4 --output-directory F:\AI_Pipeline\Outputs
```

这是优化版下次启动使用的参数，工作流本身只依赖原生节点。`--cache-ram 4` 启用按内存压力管理的节点缓存，活动缓存保留约 4 GB 系统可用内存余量；它不是固定 4 GB 缓存大小。未变化的输入和编码可复用，修改提示词会使对应编码重新执行。本轮没有重启服务，已运行的 `--cache-none` 进程不会因修改文档自动切换配置。批量任务串行排队，视频生成与关节训练错开，整批结束后再释放模型及缓存。

## 运行后释放资源

只需释放模型与缓存时，使用 ComfyUI 原生 `/free` 接口，服务继续保留。先读取队列，确认没有运行或等待任务：

```powershell
$q = Invoke-RestMethod 'http://127.0.0.1:8188/queue'
$q | ConvertTo-Json -Depth 4
```

确认 `queue_running` 和 `queue_pending` 都为空后，单独执行：

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8188/free' -Method Post -ContentType 'application/json' -Body '{"unload_models":true,"free_memory":true}'
```

接口只设置后台释放标志，返回成功不代表释放已完成。稍候再读取 `/system_stats`，检查 `devices` 中的 `torch_vram_total`，并确认服务仍可响应。系统总显存还包含其它程序的占用，不应要求它归零；暂未释放时先确认后台处理状态，不直接终止或重启服务。释放、停止、启动和 Git 操作分别执行，不放在同一条复合命令中。

## 已配置参数

| 项目 | 设置 |
| --- | --- |
| 主模型 | `minimax_h3_fl2va_pruned_w6a8.safetensors` |
| 文本与图像编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`，类型 `minimax` |
| 视频 VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| 音频输出 | 不加载音频 VAE，不解码或保存音轨；H3 内部联合采样仍保留 |
| 输出 | 1024×576，24 fps，192 帧，即 8 秒 |
| 采样 | `res_multistep` + `simple`，16 步，denoise 1，BasicGuider（CFG 1） |
| 种子 | 2026102701，固定 |
| 图像约束 | 仅连接现有标定首帧，尾帧留空，动作结束时保持自然护卫 |
| 视频解码 | 原生分块解码，512 像素空间块、64 帧时间块 |
| 保存 | MP4 / H.264，CRF 18 |

1024×576 与原始 1920×1080 首帧严格同为 16:9，输出的宽、高均为 32 的倍数。原生首帧节点会缩放首图；这里不改变宽高比、不移动中线，也不分别裁剪两栏。8 秒对应 H3 的合法帧数 `17×11+5=192`。

## 视角与关节对应

骨架、首帧和相机均来自当前训练工作包。左栏是从世界坐标 -Y 方向观察的正面机位；右栏相机位于 +X，观察的是角色的**解剖学左侧**，起始人物朝画面左方。“右栏”不是“人物右侧”。两相机等高、等比例、相差九十度；首帧中的两个视角来自同一副骨架。

提示词只保留双视角同步、固定镜头、全身完整清晰及稳定背景的必要画面约束；动作以沉身蓄力、踏步格挡、转髋出拳、提膝进攻和落步收势自然衔接，强调重心、惯性和快慢变化，不按秒或逐关节指定轨迹。结尾自然保持戒备，不强制回到最初站姿。

首帧锚定和文字约束不能把 H3 的中间帧变成严格三维投影。若需要逐帧严格同骨架，应使用同一三维动作的双视角渲染或逐帧控制；当前流程的中间帧同步程度需要看生成结果。

历史基线试跑（优化前，20 步、同图首尾约束、有音轨）已完成：RTX 3090，1024×576、192 帧、24 fps，端到端约 14 分 10 秒。抽查 0～8 秒关键画面，固定分栏与机位保持，出拳、提膝、回位基本同步；中间帧仍有少量关节位置偏差。视频文件为 `H3_ComplexAction_00001_.mp4`，本地查看图位于 `work/MedievalGirlTraining/Review/H3_ComplexAction.png`。

## 本轮调整边界

采样由 20 步降为 16 步，保留 `res_multistep`、`simple`、固定种子和完整八秒组合。去掉音频 VAE 与解码节点，只节省音频后处理开销，不代表 H3 采样变为纯视频模型。

提示词沿用参考角色外观，强调稳定背景、全身清晰及两视角同步。现有首图与 `cameras.json` 配套，保持原分辨率和构图，不拉伸、裁切或改变人物占幅；重新构图或更换动作首帧应配套重新渲染并核对标定，本轮不生成图片。

不添加放大、插帧、实验性注意力、Turbo LoRA、多帧参考或新的识别器配置；它们分别需要额外验证、模型或训练。自然动作版使用已确认的新提示词、16 步、仅首帧约束及无音轨输出，与 20 步历史视频分别保存。提示词保留用户确认的声音描述，但工作流不解码或保存音轨。历史视频仅用于基线比较，不代表当前参数效果。

自然动作版已实际生成：`H3_NaturalAction_00001_.mp4`，1024×576、192 帧、24 fps，16 步，耗时 609.69 秒（约 10 分 10 秒）。抽查九个时刻，全身与固定分栏保持，但后半段多帧停留在出拳姿态，未清楚呈现完整提膝和收势；自然连贯的动作目标未通过。本地查看图为 `work/MedievalGirlTraining/Review/H3_NaturalAction.png`，实际参数及结果记录为 `work/MedievalGirlTraining/Input/MiniMaxH3/generation-natural-record.json`。

## 官方依据

- [MiniMax 官方基础模式提示词指南](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)：首尾帧对应说明、单镜头连续动作、三个提示词字段。
- [ComfyUI H3 提示词指南](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-prompt-guide)：正向表达、BasicGuider 不使用负面提示词分支。
- [ComfyUI H3 使用指南](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)：W6A8、帧数规则及基础采样设置。

官方文字示例推荐英文；本项目按中文提示词要求撰写，保留 `Picture`、`Shot` 和三个字段名。没有另设负面提示词，也没有改动 ComfyUI 或 video2rig 核心代码。
