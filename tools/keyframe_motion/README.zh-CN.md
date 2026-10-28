# 关键姿势与动作准备工具

项目制作与验收规则由 gameart 的 `Docs/Workflows/KeyframeMotion.md` 维护；本仓库的 [技术约定](../../docs/MOTION-KEYFRAMES.zh-CN.md) 负责帧数、输入接口及能力边界。这是原版之外的输入和输出工具，不修改核心、不提供界面、不调用视频生成接口。依赖复用 Blender 的 Python/numpy，以及原 video2rig 环境的 numpy/scipy。

## 已有入口

| 入口 | 作用 |
| --- | --- |
| `blender_keyframes.py init` | 固定角色、骨架静止矩阵、源文件和相机基准 |
| `capture` | 从当前 Blender 文件的指定帧捕获姿势、手型、根变换、接触骨骼；可保存附近帧 |
| `render` | 使用基准相机渲染单姿势正侧面 PNG，不随姿势重新取景 |
| `materialize` | 将共享姿势应用到处理骨架，另存可编辑/可用于训练的 Blender 源 |
| `transfer` | 把共享姿势或原版拟合 NPZ 按静止矩阵转回原骨架，保存 Blender Action |
| `motion.py prepare` | 准备动作输入、配套起始骨架、提示词与参数；引用共享工作流，不排队 |
| `motion.py request` | 把指定动作参数填入共享 API 图，输出一次性请求；不排队、不保存工作流副本 |
| `motion.py join` | 检查共享边界合同；提供两条实际 NPZ 后报告位置、旋转和速度差 |

当前骨轴转换支持本项目已记录的同名、同层级骨架；被移除的骨必须是原处理记录中的无权重叶骨。不是任意角色的自动重定向。写入目标骨架不能带驱动或约束；有控制器的制作源可先捕获其求值姿势，再应用到无约束的处理副本。NPZ 回传目前拒绝带独立道具轨道的动画，不静默丢弃道具。

## 命令

在仓库根目录执行。以下 `blender` 和 `python` 替换为现有程序路径；`LIB`、`PROJECT`、文件路径是用户工作目录。Blender 使用 `--python-exit-code 1`，防止 Python 失败仍被当作成功。

```powershell
# 1. 用标定时的处理模型建立基准；PROJECT 是已有训练项目。
blender --background processed.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- init LIB --project PROJECT --id MyCharacter

# 2. 在 Blender 摆好动作姿势或选好动画帧，保存源文件后捕获。
blender --background authored.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- capture LIB --id GuardLeftReady --frame 1 --contacts foot_l foot_r
# 运动中的连接状态可加 --window 2，保存前后各两帧；静态姿势不虚构速度。

# 3. 渲染固定相机关键图。处理模型须与 init 完全一致。
blender --background processed.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- render LIB --id GuardLeftReady --samples 16

# 4. 编辑动作合同，然后只准备文件。
python tools/keyframe_motion/motion.py prepare LIB tools/keyframe_motion/examples/RightPunch.json
# 可选 --comfy-input <现有ComfyUI/input目录>，只安装输入 PNG，不提交任务。
```

姿势 ID 使用 PascalCase；引用后的姿势不覆盖。需要修改时捕获为新 ID，更新受影响动作合同并准备新的动作包 ID。渲染器检查实际网格是否出框并记录地面高度，但不把数值检查当成姿势视觉验收。

动作合同示例：[直拳](examples/RightPunch.json)、[高位格挡](examples/HighBlock.json)。两者共享 `GuardLeftReady`，实际片长分别为 56/24≈2.33 秒、39/24≈1.63 秒。模型原生帧数匹配最近合法值，不通过改播放速度或补帧达到时长。

## 一套共享工作流与动作参数

只维护 `workflows/keyframes/Action.workflow.json` 一套界面节点图及同图的 `Action.api.json`。界面版可导入 ComfyUI，默认展示直拳样例；切换动作修改首尾图片、提示词、帧数、步数、种子和输出前缀，不另存动作工作流。API 版由下列命令自动填参，批量任务逐个传入动作包即可，所有动作复用同一图：

```powershell
python tools/keyframe_motion/motion.py request LIB/jobs/RightPunch
# 可选 --out <不存在的请求文件.json>；仅导出一次性请求，不提交队列。
```

`request` 的标准输出为完整 API JSON，实际帧数与时长写到标准错误；可供获授权的执行器直接读取，无需落盘。界面手动操作时，节点输入应与该请求一致，包括工具计算的合法帧数和带实际时间的完整提示词。请求文件是一次执行的材料，不作为第二份长期维护的工作流。

每个动作包只维护以下数据，修改参数后再次执行 `request` 即生效：

| 文件 | 用途 |
| --- | --- |
| `parameters.json` | 当前图像名、短长类别、目标时长、分辨率、步数、种子和输出前缀 |
| `prompt.txt` | 当前动作与节奏正文；请求时自动补充实际片长、尾帧时间和输入图对应关系 |
| `action.json` | 准备时的动作合同和首尾姿势 ID；保留起始依据，不覆盖当前参数或提示词 |
| `prepared.json` | 角色和首尾指纹、来源、共享模板的相对路径及识别器准备状态 |
| `start/`、`rig/`、`comfy-input/` | 首尾图、相机和起始骨架，以及供 ComfyUI 加载的输入图 |

只改节奏、时长或采样参数时编辑当前 `prompt.txt` 与 `parameters.json`，不重建角色和姿势库；目标时长变化后须核对正文中的节奏时间。换首尾姿势时重新准备配套动作包，不能只改图片而沿用旧 `rig.json`。分辨率保持标定比例，输入图须与包内首尾图一致。

先把 `comfy-input/` 的 PNG 安装到现有 ComfyUI input（或在 prepare 时指定该目录），再由明确的生成操作执行。模型与节点默认值只在共享图维护；当前默认 W6A8 H3、16 步、1024×576、24 fps、首尾帧约束，无放大、插帧或音频解码。变更共享图时同步界面/API 格式与安装副本，并核对现有动作参数；仅改动作时不改图结构。

包内 `rig/rig.json` 的 `start_pose` 和物体变换来自实际首姿势，`start/start.png`、`start/end.png`、`start/cameras.json` 与其配套。中间姿势暂不接入；合同含额外字段时会报错，避免写了中间约束却实际没有使用。

视频生成后另存到 `JOB/clips/<动作ID>.mp4`。**准备包不自动复制 reader.pt**：原训练项目当前尚未完成新版 reader 训练；即使以后有权重，换首姿势后的复用也需要检查，而不是仅凭文件存在决定。未准备识别器时不要直接运行原版 `run`，它会进入旧跟踪路线。

若为新起点训练，先保存正确的处理骨架姿势源：

```powershell
blender --background processed.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- materialize LIB --id GuardLeftReady --out JOB/Input/start.blend
# 以下是后续单独授权执行的原版步骤，不由本工具自动运行：
# 先按人形输入流程生成 JOB 的配套训练数据，再调用训练器：
# python -m video2rig.reader.train JOB --steps 9000
# python -m video2rig check JOB RightPunch
# python -m video2rig run JOB RightPunch
# python -m video2rig review JOB RightPunch
```

原版 prepare 的随机训练姿势默认没有本 fork 已确认的人形与手型约束；正式重建数据时应按 [人形输入流程](../humanoid_inputs/README.zh-CN.md) 生成配套姿态、图片与遮挡，再调用原版训练器。不要为方便训练退回无约束随机姿势。

## 原骨架回传与衔接

```powershell
# 只回传一个共享姿势。
blender --background original.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- transfer LIB --pose GuardLeftReady --out guard-original.blend

# 拟合完成后回传真正动作；JOB 的起始数据用于保留未拟合的骨骼姿势。
blender --background original.blend --python-exit-code 1 --python tools/keyframe_motion/blender_keyframes.py -- transfer LIB --motion JOB/out/RightPunch.npz --job JOB --out punch-original.blend

# 只检查两条动作的共享端点合同。
python tools/keyframe_motion/motion.py join LIB/jobs/RightPunch LIB/jobs/HighBlock
# 实际动画产生后追加：--motion-first <第一条.npz> --motion-second <第二条.npz>
```

转换公式为 `P_original = (P_processing × inverse(R_processing)) × R_original`，再按原父子层级求局部姿势。它保留蒙皮变换，不直接复制不同骨轴的四元数。回传记录实际写入矩阵误差，保留原骨架；不输出 FBX，不写入 Unity。

`join` 的实际动作检查不自动对齐根位置、不修复脚滑、不自动判定自然程度。报告全部记录骨骼头部的位置差、姿态角差及相邻帧线速度/角速度差；共享静态姿势通过只代表连接合同一致。运动中的边界窗口是制作依据，当前 H3 首尾图入口不消费窗口视频，不能把保存了窗口称为已经约束了速度。

## 当前本机样例

本地姿势与角色文件位于 `work/MedievalGirlKeyframes/`，不随 Git 分发。`Source/Guard.blend` 是可编辑护卫姿势源，`poses/GuardLeftReady/frame.png` 为固定相机双视图。此样例用于贯通基础设施，不代表完整动作或游戏验收。本机 ComfyUI 的 Video2Rig 工作流目录只安装共享入口 `Keyframes_Action.json`；直拳和格挡通过各自的 `parameters.json` 与 `prompt.txt` 填入同一图，输入图已配套安装，没有提交生成。

实际检查记录在该工作包的 `infrastructure-record.json`。角色回传已用保存后重新打开的单姿势验证；新拟合动作的 NPZ 回传分支与跨片动态速度，须在有实际动作后继续验证，不能用静态共享姿势的通过结果替代。

创建姿势、动作包、回传文件以及通过 `request --out` 导出的请求均拒绝覆盖已有目标；动作包中的当前参数和提示词可以正常编辑。失败可能留下未完成目录；应先检查具体原因与任务归属，再按正常文件清理规则处理，不用强制重试掩盖失败。
