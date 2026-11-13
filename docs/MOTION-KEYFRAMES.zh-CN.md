# 关键姿势工具的技术约定

制作与验收条件由 gameart 的 `Docs/Workflows/KeyframeMotion.md` 维护。本页定义本地固定相机入口的接口和限制；具体命令见[关键姿势工具](../tools/keyframe_motion/README.zh-CN.md)。以下与 MedievalGirl 实跑时的 fork 接口一致；更换工具、模型或节点后以实际源码和输出复核。

## 固定相机入口的前提

- **骨架源。** `init` 打开与 `PROJECT/rig/rig.json` 相同姿势和物体矩阵的处理版 `.blend`，读取 `PROJECT/Input/preparation.json` 中的 `original_rest_matrices` 和 `removed_unweighted_leaf_bones`。前者是仍参与转换的骨名及原骨架静止矩阵，后者是处理副本删掉的无权重叶骨列表。对应骨须同名、同父级并使用支持的继承方式。其他角色先制作并核对自己的处理记录；`prepare_humanoid.py` 按 MedievalGirl 的实际骨名、辅助骨和单位编写。
- **相机。** `cameras.json` 的 `views` 依次是 `v2r_front` 与 `v2r_side`，等高等尺度的正交投影，视线相差九十度，画幅与分栏连续。实际机位及解剖左右由相机决定。固定入口不会逐帧缩放、平移或重标定。
- **关节标定。** `creature.json` 记录根骨 `root_bone`、世界朝向 `facing` 及 `hinges` 的实际骨名和 `front/back` 弯曲方向，不能沿用原版 `auto` 代替明确标定。物体只支持正向均匀缩放；`calibrate` 从中性源生成米制换算和固定铰链。当前按人形给出的肘部 150°、膝部 140°、伸直侧余量 2°是工具的初始范围；新骨架须先核对。超范围的首姿势须纠正基准或姿势，不能靠裁剪混过去。
- **范围。** `prepare` 不支持非根骨平移或姿势中的非刚性缩放；它能记录 `root_motion` 合同，但当前固定 `read/fit` 只接受 `in_place`，不拟合独立道具轨道。写回源有驱动或骨骼约束时，应在配套的可烘焙变形骨架副本上写 Action。改过骨轴的处理版不能把局部四元数直接复制到原骨架；`transfer` 按变形矩阵 `P_original = (P_processing × inverse(R_processing)) × R_original` 重算原骨架局部姿势。
- **直接拟合原骨架。** 两套静止矩阵相同的情况下，固定入口先按处理骨架检查，`transfer` 可能把原骨架当成处理骨架或因额外辅助骨报错。无需改轴的绑定可走原版同骨架的拟合与导入路线；不要为了通过 `transfer` 人为改轴。

## 共享生成图和时间

[Action.workflow.json](../workflows/keyframes/Action.workflow.json) 与 [Action.api.json](../workflows/keyframes/Action.api.json) 是同一节点图的界面和 API 格式。已验证的配置为 MiniMax H3 W6A8、16 步、1024×576、24 fps，使用首尾图；无放大、补帧或音轨输出。默认值以共享图为准，模型与节点见 [ComfyUI H3 指南](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)。ComfyUI 的安装副本可沿用 `Keyframes_Action.json`，修改节点后同步两个源格式和部署副本。新增动作只新增动作数据。

| 文件 | 职责 |
| --- | --- |
| `action.json` | 准备时的动作定义和首尾姿势 ID；保存起始依据。 |
| `prepared.json` | 角色、首尾指纹及工作流引用；`start/`、`rig/`、`comfy-input/` 保存配套相机、起始骨架和图。 |
| `parameters.json`、`prompt.txt` | 当前生成输入。重新 `request` 时读取实际图片名、时长、画幅、采样参数和提示词；保存当次完整请求以便追溯。 |

换首姿势要同时换图片和对应的 `rig.json`。现有图没有接入中间关键图和边界窗口；即使某个生成器支持中间引导，仍须有实际接线、时点和输出记录才能认定约束生效。

H3 的此入口按 24 fps、`17k+5` 帧选最接近目标的合法帧数：短片为 39 / 56 帧（约 1.63 / 2.33 秒），长片为 73 / 90 / 107 帧（约 3.04 / 3.75 / 4.46 秒）。画幅保持标定宽高比，宽高须是 32 的倍数。视频时长是帧数除以帧率，末帧时刻是 `(帧数 − 1) / 帧率`；`request` 会写出两者，动作正文的节奏由制作者核对。固定 `fit` 按帧号除以 24 求解并检查尺寸、帧数，但不会据此证明原视频实际帧率；入库前另用媒体信息确认。

## 识别与拟合的数据

- 角色级训练集的姿态、分栏样图、标签、遮挡、骨架和相机须配套，权重与完整关节定义绑定。固定 `read` 核对标签定义，使用 8 次平移缩放采样；热图峰值用于观测强弱，不是正确率。识别记录保存视频、相机、骨架、权重和标定等文件指纹。
- 固定 `fit` 将网格、静止矩阵平移、骨长和空间标签统一到米，保留相机的世界投影；按固定相机与实际帧号拟合。启用本 fork 的逐帧求解、左右纠错及可用观测筛选，关闭自动由脚步推算行进与双栏漂移补偿。IK 辅助点不作为有效观测。记录原始及实际使用的点，导出 NPZ 时换回处理骨架单位；不能把 `fitted_not_accepted` 当成动画验收。
- 上游在 [`fd3ef75`](https://github.com/canitcode/video2rig/commit/fd3ef75) 引入整段联合拟合和按共同投影方向匹配两栏时间。当前固定入口没有接入该算法；原版入口还可能做轮廓与首帧相机匹配、估算内容帧率。选择另一入口后，按它实际使用的相机、时序与观测重新检查。
- `join` 检查共享边界；提供实际拟合 NPZ 后增加位置、角度和相邻帧速度报告。它不修动作，不读取 Blender 修整结果。Blender 成品、FBX 和 Unity 各自回读。
