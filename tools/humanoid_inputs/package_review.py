"""用原版读取器检查数据配套，并输出人看的 PNG 预览；预览不作为训练图片。"""
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

if len(sys.argv) != 2:
    raise SystemExit('Usage: python ' + Path(__file__).name + ' PROJECT')
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video2rig.common import Project
from video2rig.creature import Creature
from video2rig.reader import keypoints
from video2rig.reader.train import load_items, load_pictures

pr = Project(ROOT)
cr = Creature(pr, pr.clip_cfg(''))
kps = keypoints(cr)
renders = ROOT / 'reader' / 'renders'
items = load_items(pr, kps, renders, cr=cr)
pictures, uv, bg = load_pictures(items)
assert pictures.shape == (48, 448, 448, 3)
assert len(items) == 48 and len({i['pose'] for i in items}) == 24
lab = np.load(renders / 'labels.npz')
poses = np.load(ROOT / 'reader' / 'poses.npz')['params']
order = [cr.rig.names.index(str(n)) for n in lab['names']]
geometry_cr = cr
hand_context = ROOT / 'reader' / 'hand_visibility'
if (hand_context / 'poses.npz').exists():
    hp = Project(hand_context)
    geometry_cr = Creature(hp, hp.clip_cfg(''))
    poses = np.load(hand_context / 'poses.npz')['params']
head_error = max(float(np.max(np.linalg.norm(geometry_cr.rig.bone_points(p)[0][order] - lab['heads'][i], axis=1))) for i, p in enumerate(poses))
assert head_error < .001, head_error
out = ROOT / 'Review'
out.mkdir(exist_ok=True)
font = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 21)
small = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 16)
lookup = {(it['pose'], it['view']): i for i, it in enumerate(items)}
main_names = ['pelvis', 'spine_01', 'spine_02', 'spine_03', 'neck_01', 'head'] + [n + s for s in ('l', 'r') for n in ('clavicle_', 'upperarm_', 'lowerarm_', 'hand_', 'thigh_', 'calf_', 'foot_', 'ball_')]
by_label = {str(k[0]): j for j, k in enumerate(kps)}
for pose in range(24):
    pair = Image.new('RGB', (896, 492), '#eeeeee')
    annotation = pair.copy()
    for view in range(2):
        i = lookup[pose, view]
        im = Image.fromarray(cv2.cvtColor(pictures[i], cv2.COLOR_BGR2RGB))
        pair.paste(im, (448 * view, 44)); annotation.paste(im, (448 * view, 44))
        draw = ImageDraw.Draw(annotation)
        for name in main_names:
            if name not in by_label:
                continue
            k = by_label[name]; x, y = uv[i, k]; x += 448 * view; y += 44
            colour = '#f5a623' if name.endswith('_l') else '#00b9ff' if name.endswith('_r') else '#ffffff'
            visible = bool(items[i]['seen'][k] and items[i]['inside'][k])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colour if visible else None, outline=colour, width=1)
    for im in (pair, annotation):
        draw = ImageDraw.Draw(im)
        category = '起始站姿' if pose == 0 else '温和 · 接地' if pose < 9 else '中等幅度' if pose < 17 else '大幅 · 动态'
        draw.text((15, 8), f'{pose + 1:02d}  {category} · 正面 / 侧面', font=font, fill='#202020')
    pair.save(out / f'pose-{pose + 1:02d}.png')
    annotation.save(out / f'pose-{pose + 1:02d}-joints.png')

# 每页 6 对，保留姿态编号；整页所有图都来自实际训练读取结果。
for page in range(4):
    sheet = Image.new('RGB', (1808, 1552), '#f5f5f5')
    d = ImageDraw.Draw(sheet)
    d.text((16, 10), f'训练输入审阅 {page + 1}/4  |  左正面、右侧面  |  原始读取为每视角 448×448', font=font, fill='#202020')
    for tile in range(6):
        im = Image.open(out / f'pose-{page * 6 + tile + 1:02d}.png')
        sheet.paste(im, (8 + tile % 2 * 900, 52 + tile // 2 * 500))
    sheet.save(out / f'contact-{page + 1:02d}.png')
record = {'stage': 'awaiting_user_review', 'poses': 24, 'views': 2, 'loaded_tensor_shape': list(pictures.shape), 'raw_format': 'JPEG RGB quality=100', 'render_fit': 896, 'network_size': 448, 'normalisation': 'original reader: BGR to RGB then ImageNet mean/std', 'labels': 'reader/renders/labels.npz', 'visibility': 'reader/visible.npz', 'keypoints': len(kps), 'labels_max_head_error_armature_units': head_error, 'no_training_started': not (ROOT / 'reader' / 'reader.pt').exists(), 'preview_note': 'PNG 为实际加载后的训练输入及独立标注预览；标注图不参与训练。左右视角按同一 pose 分组，完整训练时也应按 pose 划分训练/验证，不能按图片随机拆分。'}
(ROOT / 'review-record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, ensure_ascii=False), flush=True)
