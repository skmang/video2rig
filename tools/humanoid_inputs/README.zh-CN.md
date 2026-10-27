# 人形输入与训练样图审阅

本目录是原版流程之外的输入准备脚本，不修改 `video2rig/`、`blender/` 或依赖要求。当前只验证过带 Unreal 风格命名的 MedievalGirl 角色；不是任意骨架的通用转换器。角色模型、贴图、生成视频和训练权重不随代码分发。

先用 24 个姿态审阅关节方向、姿态幅度、手型和主体占比；确认后可按同样约束重新生成 3,000 个训练姿态。图片准备与识别器训练是独立步骤，脚本不会自动启动训练。

## 处理内容与边界

- `prepare_humanoid.py` 在处理副本中删除八根已确认无权重、无子骨的 twist 辅助骨，校正主骨朝向，并轻微外展双腿。源文件须为已打包贴图、无动画的 Blender 模型；骨名、单位和结构须与该示例一致。脚本拒绝带骨骼约束或对象驱动的输入，并比较校正前后的中性网格。
- **骨轴会改变。** 原始素材保持不动；处理副本的动作不能直接按局部四元数复制回原游戏骨架。`Input/preparation.json` 保存两套静止矩阵，后续需要按它们转换旋转。目前没有实现游戏骨架回传工具。
- `prepare_review_poses.py` 调用原版姿态采样器，收紧人形关节活动范围，过滤明显的腿部交叉、手臂穿入躯干和出框候选，再按温和、中等、大幅三档选择差异化姿态。默认 24 个，`--poses 3000` 生成完整训练规模；`--seed` 控制重建。完整批次会打散幅度档，使原版末尾 5% 验证集也覆盖各档。躯干检查是近似几何检查，不是完整人体碰撞或物理平衡求解。
- `prepare_hand_variations.py` 增加自然放松、张掌、半握、抓握、握拳五类手型。手型用于外观变化，**原版仍不训练逐根手指的识别和动作拟合**。它另外建立 `reader/hand_visibility` 上下文，让原版遮挡计算使用实际变化后的手指几何；不能用主项目的默认遮挡命令覆盖这份结果。
- `render_review.py` 使用 Blender 公开渲染钩子把 JPEG 质量设为 100，其余渲染和标签生成继续调用原版脚本。
- `package_review.py` 通过原版读取器加载全部训练输入，检查关节标签、遮挡数组及包含实际手型的网格是否出框，从全批次抽取 24 个姿态输出正侧面 PNG 和标签预览。PNG 预览不作为训练输入。

## 数据格式

原版训练器按 `视角/批次.jpg` 和 `labels.npz` 中的槽位读取图片。每个视角最终等比补边至 448×448，再执行原版 RGB/ImageNet 归一化。不能只改图片后缀或独立裁切图片而不更新相机与标签。

本示例以 `--fit 896` 渲染，再由原读取器缩小到 448；JPEG 使用 RGB、质量 100。首帧相机 `--margin 1.35` 时，站立角色约占输入高度 74%，而不是先前的大留白。采样会拒绝超出相机安全边界的姿态；动作范围改变时，应重新评估相机，而不是裁断四肢。

`reader/poses.npz`、`reader/renders/labels.npz`、`reader/visible.npz` 和 `start/cameras.json` 必须配套。同一姿态的不同视角应一起划分到训练或验证集；不能按图片随机拆分。更新同数量姿态后，要移除本数据集生成的旧 `reader/renders/params.npy`，原版缓存只核对形状，无法识别内容已经变化。

## 运行顺序

在仓库根目录执行。下面用 `python` 和 `blender` 表示独立环境及 Blender 可执行文件；Windows 可替换为各自的完整路径。`work/my_character` 是本地工作目录，先复制 `creature.example.json` 为其中的 `creature.json`。所有写入均位于该工作目录。

```powershell
blender source.blend -b --python tools/humanoid_inputs/prepare_humanoid.py -- work/my_character
blender work/my_character/Input/MedievalGirl_Humanoid.blend -b --python blender/export_rig.py -- --out work/my_character/rig
blender work/my_character/Input/MedievalGirl_Humanoid.blend -b --python blender/render_start.py -- --out work/my_character/start --facing=-Y --margin 1.35 --samples 64
python tools/humanoid_inputs/prepare_review_poses.py work/my_character
python tools/humanoid_inputs/prepare_hand_variations.py work/my_character
blender work/my_character/Input/MedievalGirl_Humanoid.blend -b --python tools/humanoid_inputs/render_review.py -- --poses work/my_character/reader/poses.npz --cameras work/my_character/start/cameras.json --out work/my_character/reader/renders --copies 4 --samples 64 --height 1080 --fit 896 --sharp --project work/my_character
python -m video2rig.reader.visible work/my_character/reader/hand_visibility --workers 2 --poses work/my_character/reader/hand_visibility/poses.npz --out work/my_character/reader/visible.npz
python tools/humanoid_inputs/package_review.py work/my_character
```

打开 `Review/contact-01.png` 至 `contact-04.png` 查看全身图；手型变化直接体现在全身训练姿态中，不另产出手部图。`Input/sampling.json` 和 `Input/hand-poses.json` 保存选择及手型记录。预览排版默认使用 Windows 微软雅黑字体；其它系统须调整 `package_review.py` 的字体路径。

完整数据使用新的工作目录，保留已确认的 `Input` 模型、`rig`、`start` 和 `creature.json`，将上面姿态采样命令改为：

```powershell
python tools/humanoid_inputs/prepare_review_poses.py work/my_character --poses 3000 --seed 20261028
```

随后依次执行手型写入、渲染、遮挡计算和数据审阅。3,000 个姿态对应 6,000 张双视角输入，磁盘按原版条带格式存放，不能按 JPEG 文件数判断样本数。保持 `--fit 896 --samples 64 --sharp` 与 JPEG 质量 100；训练时由原版执行模糊、颜色、分辨率和噪声增强。

此流程只准备图片，不自动启动训练。不要直接运行原版 `reader.prepare` 覆盖这里的姿态和遮挡结果。相机、姿态或骨架改变后，需要重建关联图片、标签和缓存。`review-record.json` 记录全量读取和标签核对结果，`Input/sampling.json` 记录姿态来源与类别。

## MiniMax H3 模型下载

独立下载器使用标准库，无需新增 Python 包。四个文件为 FL2VA W6A8 主模型、NVFP4 文本编码器及视频/音频 VAE，合计约 35.1 GB。固定 Hugging Face 版本并检查官方 SHA256；下载分段可续传，校验后才放入模型目录。首次下载和合并需要为分段及成品预留约两倍空间。

```powershell
python tools/download_minimax_h3.py --models-dir PATH_TO_COMFYUI/models --cache-dir downloads/minimax-h3
```

下载器拒绝覆盖同名但校验不符的现有模型，也不安装或升级 ComfyUI。不要同时对同一缓存目录启动两个下载进程。`download-manifest.json` 仅记录已经校验完成的文件；分段缓存可在全部校验完成后清理。

来源：[Comfy-Org 模型](https://huggingface.co/Comfy-Org/MiniMax-H3)、[ComfyUI H3 文档](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)。权重适用 H3 自身许可，仓库的 AGPL 不替代模型许可。本流程尚未据此验证 H3 双视角生成质量。

## 提交范围

提交脚本、配置示例和必要说明；本地工作包留在 `work/`。模型权重、角色源资产、训练图片/标签、缓存、下载分段和本地凭据均不提交。已放弃的网页演示原型不再保留在当前目录，仍可从 Git 历史查看。
