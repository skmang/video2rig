# 关键姿势与动作准备工具

制作顺序、何时可复用权重以及每步的验收条件由 gameart 的 `Docs/Workflows/KeyframeMotion.md` 维护。本页给出实际命令；[技术约定](../../docs/MOTION-KEYFRAMES.zh-CN.md)列出相机、帧数、数据配套及工具能力。

## 先选入口

- **固定相机入口：** `tools/keyframe_motion/`，已实跑 MedievalGirl 原地动作。使用配套的处理骨架、原骨架和固定正侧面相机。不同人形骨架须先证明对应关系及关节标定可用。要处理行进根运动、独立道具轨道、侧面加顶视或不同骨架间的重定向，不能直接用这套命令。
- **原版入口：** 不满足固定入口前提时，从[本 fork 原版流程](../../docs/WORKFLOW.md)或选定的[上游版本](https://github.com/canitcode/video2rig/blob/2b94937/docs/WORKFLOW.md)开始。它们各自处理相机、时序、拟合和导入。按同一版本使用 `check/run/review` 与 `blender/import_motion.py`，先确认该项目下有配套 `reader/reader.pt`，否则 `run` 会走旧点跟踪。新物种或新绑定先做一条短片验证；按实际结果判断能否用于生产。

以下命令只用于**固定相机入口**，在 video2rig 仓库根目录执行。例子用 PowerShell；`blender` 和 `python` 表示实际 Blender 与 video2rig Python 环境的程序，可换成完整路径。把 `PROJECT`、`LIB` 等占位符换成实际位置，含空格的路径加引号。Windows 上将训练项目、姿势库和处理版 `.blend` 放在同一盘符，工具会记录它们之间的相对路径。运行前准备可用的 CUDA PyTorch、本地 DINOv2-small 缓存和已安装 H3 模型的 ComfyUI。依赖见[仓库说明](../../README.md#what-you-need)。渲染、训练、H3 生成和识别共享显卡时按顺序执行。

## 1. 新角色：准备训练项目与姿势库

```text
PROJECT/                      角色训练项目
  Input/preparation.json       原骨架对应关系
  creature.json                根骨、朝向、关节配置
  rig/                         处理骨架与网格
  start/cameras.json            固定相机
  reader/                      姿态、样图、标签、遮挡、共享权重

LIB/                          与 PROJECT 关联的角色姿势库
  character.json, fitting.json  角色与拟合基准
  poses/<姿势ID>/                三维姿势与固定相机图
  jobs/<动作ID>/                 各动作的输入、观测与拟合结果
```

1. 在 `PROJECT` 准备中性的处理版 Blender 文件、原角色源和骨架处理记录。对与 MedievalGirl 相容的输入可按[人形样图流程](../humanoid_inputs/README.zh-CN.md#运行顺序)执行。其它绑定先按[技术约定的输入要求](../../docs/MOTION-KEYFRAMES.zh-CN.md#固定相机入口的前提)核对源及对应关系；`init` 不会替你生成映射。
2. 用同一份处理版导出 `PROJECT/rig/` 和 `PROJECT/start/cameras.json`。相机留出动作活动空间，正面在左栏，侧面在右栏。以下 `-Y` 与左侧机位只是示例；按实际朝向选择，用 `--margin` 调整两栏的留白。已有一致的训练项目可直接从 `init` 开始。

```powershell
blender --background processed.blend --python-exit-code 1 --python blender/export_rig.py -- --out PROJECT/rig
blender --background processed.blend --python-exit-code 1 --python blender/render_start.py -- --out PROJECT/start --views front,side --facing=-Y --side left
blender --background processed.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- init LIB --project PROJECT --id MyCharacter
python tools/keyframe_motion/motion.py calibrate LIB
```

`init` 写出 `LIB/character.json`，`calibrate` 从固定的中性骨架写出 `LIB/fitting.json`。后者包含米制换算和关节轴、弯曲方向及范围。用实际屈伸、接触和回传检查标定，不能从每条动作的护卫姿势重新猜关节方向。

3. 为动作摆姿势或从已有动作取样，保存为 `authored.blend`；它可用配套原骨架或处理骨架。以下骨名和帧号换成实际的。另一个端点重复捕获和渲染；首尾相同就复用同一个 ID。

```powershell
blender --background authored.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- capture LIB --id GuardReady --frame 1 --contacts foot_l foot_r
blender --background processed.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- render LIB --id GuardReady --samples 16
```

`capture` 保存 `pose.json`，`render` 用绑定在 `character.json` 的处理源和相机生成 `frame.png`、`render.json`。多个骨架用 `--armature` 指定；运动边界可给 `capture` 加 `--window 2` 保存前后两帧。需单独保存处理版的该姿势时，用 `materialize LIB --id GuardReady --out guard-processed.blend`，仍打开同一处理源。姿势 ID 使用 PascalCase；修改已被引用的姿势时换 ID。

## 2. 新角色或新观测条件：准备并检查共享识别器

识别器由角色级 `PROJECT/reader/reader.pt` 提供。固定入口从关联项目读取，不复制到每个动作。已有权重先按[配套条件](../../docs/MOTION-KEYFRAMES.zh-CN.md#识别与拟合的数据)和实际独立样本核对；配套证据有效时跳过训练。

需要重建数据时，先备好 `reader/poses.npz`、`reader/renders/` 下按视角保存的样图与 `labels.npz`、`reader/visible.npz`。MedievalGirl 人形约束见[训练样图步骤](../humanoid_inputs/README.zh-CN.md#运行顺序)；其它角色可查[原版姿态、渲染与遮挡入口](../../docs/WORKFLOW.md#2b-train-the-joint-reader-once-per-creature)，按角色核对样图。不要用 `reader.prepare` 覆盖已确认的训练数据。生成新姿态、相机或标签后不能混用旧 `params.npy`。

另备 `PROJECT/reader/independent/`。可先用与训练不同的种子抽一组姿态检查基础观测；再准备动作特有的极值、遮挡和接触姿势。以下第一条是可选的基础抽样，已有针对性样本时跳过它。针对性样本用该处理骨架的关节参数 `params`（每个姿势一行），按[姿态编码与写出示例](../../video2rig/reader/poses.py#L397)同时写出配套的 `bones`、`root`、`quats` 和 `root_location`。后两条分别用姿态渲染图像及标签、用同一批参数计算遮挡。`capture` 的单姿势 JSON 不能直接传入，两个数据集也不能混用。

```powershell
python -m video2rig.reader.poses PROJECT --n 24 --seed 20261101 --out PROJECT/reader/independent/poses.npz
blender --background processed.blend --python-exit-code 1 --python blender/render_poses.py -- --poses PROJECT/reader/independent/poses.npz --cameras PROJECT/start/cameras.json --out PROJECT/reader/independent --project PROJECT --sharp
python -m video2rig.reader.visible PROJECT --poses PROJECT/reader/independent/poses.npz --out PROJECT/reader/independent/visible.npz
```

然后分别训练、评估留出集与独立集：

```powershell
python tools/keyframe_motion/motion.py train LIB --hf-home DINO_CACHE --execute
python tools/keyframe_motion/motion.py evaluate LIB --hf-home DINO_CACHE --execute
python tools/keyframe_motion/motion.py evaluate LIB --hf-home DINO_CACHE --renders reader/independent --all-poses --execute
```

`DINO_CACHE` 是包含 `hub/` 的现有缓存根；该入口离线，不会下载。去掉 `--execute` 只查看待执行任务。训练默认 9,000 步、batch 24、seed 20261028，可按验证结果选择训练预算。`train` 写权重、训练结果和输入指纹；`evaluate` 分别为按姿态留出的约 5% 与指定独立集写出分视角、分关节报告。其像素以标定画幅计，与视频像素比较前要换算。`evaluate` 是单次识别评估，生产 `read` 用八次平移缩放采样；发现错点时用生产读法和实际拟合定位，不按平均误差或热图峰值单独判定。

## 3. 每条动作：准备输入、排队并选片

以[直拳](examples/RightPunch.json)或[高位格挡](examples/HighBlock.json)为字段示例，准备自己的 `ActionSpec.json`。其中 `id` 决定下面的 `ActionId`，首尾姿势 ID 必须已经渲染。原地拟合使用 `root_motion: "in_place"`；独立根运动不走本入口。使用实际角色关节、左右视角和动作意图。

```powershell
python tools/keyframe_motion/motion.py prepare LIB ActionSpec.json --comfy-input COMFY_INPUT_DIR
python tools/keyframe_motion/motion.py request LIB/jobs/ActionId --out REQUEST_JSON
```

`COMFY_INPUT_DIR` 是现有 ComfyUI 的输入目录；也可省略该参数，之后按原文件名安装动作包 `comfy-input/` 里的 PNG。初次生成以后只改节奏、时长或采样参数时，编辑动作包的 `prompt.txt`、`parameters.json` 再执行 `request`；换首尾姿势则重新准备动作包。自动生成的提示词含默认左侧机位，若实际拍的是右侧，先修正文字以匹配相机。保持中文动作和画面约束。每次生成保存实际请求；`REQUEST_JSON` 的父目录须存在、文件须是新路径。

`request` 只填共享图，不排队。人工在界面操作时，导入 [Action.workflow.json](../../workflows/keyframes/Action.workflow.json)，按请求更新节点输入。用 API 时，把请求 JSON 作为 `prompt` 提交给已经运行的本机 ComfyUI，例如：

```powershell
$graph = Get-Content -LiteralPath 'REQUEST_JSON' -Raw -Encoding utf8 | ConvertFrom-Json
$body = ConvertTo-Json -InputObject @{ prompt = $graph } -Depth 100 -Compress
$queued = Invoke-RestMethod 'http://127.0.0.1:8188/prompt' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
$queued.prompt_id
```

把 `REQUEST_JSON` 和地址换成实际值。记录 `prompt_id`；任务结束后，用该 ID 查 `GET /history/<prompt_id>`。从响应里取该 ID 对应的记录，确认 `status.completed=true`、`status.status_str=success`，从 `outputs` 中的 `SaveVideo` 节点查实际文件。当前图和帧数见[技术约定](../../docs/MOTION-KEYFRAMES.zh-CN.md#共享生成图和时间)。核对原片的实际帧数、分辨率和 24 fps，例如：

```powershell
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of json CANDIDATE.mp4
```

按项目流程审片，再把选定视频保存为 `LIB/jobs/ActionId/clips/ActionId.mp4`。候选及完整请求分别留档，不能按文件序号猜哪条通过。

## 4. 每条动作：识别、拟合、回传

```powershell
python tools/keyframe_motion/motion.py read LIB/jobs/ActionId --hf-home DINO_CACHE --execute
# 审阅抽帧图和叠加视频，确认左右身份和两栏动作时序，再执行 fit。
python tools/keyframe_motion/motion.py fit LIB/jobs/ActionId --execute
# 审阅实际使用的点和三维结果后，将动作回传到配套原骨架。
blender --background original.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- transfer LIB --motion LIB/jobs/ActionId/out/ActionId.npz --job LIB/jobs/ActionId --out action-on-original.blend
```

| 入口 | 产物与下一步 |
| --- | --- |
| `read` | 动作包的 `work/ActionId/` 下得到 `reader-keyframes.npz`、输入记录、`reader-keyframes-sheet.jpg`、`reader-keyframes-overlay.mp4`。核对峰值与遮挡，空心点提示峰值偏低，不代表位置一定错。 |
| `fit` | 同目录的 `observations-used.npz` 保存实际使用的点、权重、重投影与两类误差；`out/ActionId.npz` 是处理骨架单位的运动，`out/ActionId.fit.json` 记录拟合依据。输出仍待姿势与动作审阅。 |
| `transfer` | 写出原骨架上的 Blender Action 和 `.transfer.json`。原骨架源必须配套、可烘焙；动作或源文件不就绪时不能跳过映射。后续编辑、导出和 Unity 验收按 gameart 主流程。 |

固定入口不执行上游 `sync.py` 的两栏时间匹配，也不重估相机。需要辅助定位时，可把固定投影和原始读点传入 `video2rig.check.panel_misses`，查看躯干与肢体在两栏的共同投影方向差异；这不是深度真值，也不检查单栏横向误读。不要直接用原版 `check/run/review` 代替固定入口。

已存在的动作包、观测和结果不会被相应入口覆盖。重做视频并已跑过识别时，为新候选准备独立的动作包 ID，或在明确归属后移出该动作的旧观测与拟合产物，保留有用的失败证据。更换视频、相机、权重、骨架或关键参数时用当前输入重新 `read` 和 `fit`，不能绕过输入指纹检查。

## 5. 对照衔接与本机实例

同一角色的两条动作可先用 `join` 核对共享边界。若使用两份拟合 NPZ，可补查修整前的首尾姿势、相邻帧线速度和角速度：

```powershell
python tools/keyframe_motion/motion.py join LIB/jobs/FirstAction LIB/jobs/SecondAction
python tools/keyframe_motion/motion.py join LIB/jobs/FirstAction LIB/jobs/SecondAction --motion-first LIB/jobs/FirstAction/out/FirstAction.npz --motion-second LIB/jobs/SecondAction/out/SecondAction.npz
```

首尾相同和局部数值报告不能代替成品预览。Blender 修整后，重新按最终 Action 验收两段接缝；`join` 不读取修过的 `.blend`，不生成 FBX 或导入 Unity。

实跑的本地输入与记录在 `work/MedievalGirlKeyframes/`，读 `execution.json` 获取当前候选和证据。`Source/calibration_motion.py` 的 `independent()` 展示了如何从处理骨架的手部目标生成出拳、格挡与交叉遮挡的独立序列。完整样例见 gameart 的 `Animations/MedievalGirlVideoMotion/asset.md`、`delivery.md`；其中 `FinishActions.py` 的修整逻辑只适用于这两条动作。
