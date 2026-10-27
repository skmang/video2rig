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
from video2rig.reader import keypoints, project_ortho
from video2rig.reader.train import load_items, load_pictures

pr = Project(ROOT)
cr = Creature(pr, pr.clip_cfg(''))
kps = keypoints(cr)
renders = ROOT / 'reader' / 'renders'
items = load_items(pr, kps, renders, cr=cr)
pose_data = np.load(ROOT / 'reader' / 'poses.npz')
n = len(pose_data['params'])
assert len(np.unique(pose_data['params'], axis=0)) == n, 'Duplicate body poses'
assert len(items) == n * 2 and len({i['pose'] for i in items}) == n
pictures, uv, bg = load_pictures(items, workers=4)
assert pictures.shape == (n * 2, 448, 448, 3)
assert np.isfinite(uv).all()
print(f'LOADED {n} poses / {len(items)} pictures', flush=True)
lab = np.load(renders / 'labels.npz')
poses = np.load(ROOT / 'reader' / 'poses.npz')['params']
order = [cr.rig.names.index(str(n)) for n in lab['names']]
geometry_cr = cr
hand_context = ROOT / 'reader' / 'hand_visibility'
if (hand_context / 'poses.npz').exists():
    hp = Project(hand_context)
    geometry_cr = Creature(hp, hp.clip_cfg(''))
    poses = np.load(hand_context / 'poses.npz')['params']
cameras = json.loads((ROOT / 'start' / 'cameras.json').read_text(encoding='utf-8'))
head_error = 0.0
min_frame_margin = 1.0
label_index = {int(pose): i for i, pose in enumerate(lab['index'])}
for i, p in enumerate(poses):
    head_error = max(head_error, float(np.max(np.linalg.norm(geometry_cr.rig.bone_points(p)[0][order] - lab['heads'][label_index[i]], axis=1))))
    mesh = geometry_cr.rig.skin(p, geometry_cr.mesh['coords'], geometry_cr.idx, geometry_cr.wts)
    for camera in cameras['views']:
        size = np.array((camera['width_px'], camera['height_px']))
        xy = project_ortho(camera, cr.data['armature_world'], mesh, size) / size
        min_frame_margin = min(min_frame_margin, float(xy.min()), float(1 - xy.max()))
    if (i + 1) % 500 == 0:
        print(f'GEOMETRY {i + 1}/{n}', flush=True)
assert head_error < .001, head_error
assert min_frame_margin > 0, f'Mesh outside frame: margin={min_frame_margin}'
assert np.load(ROOT / 'reader' / 'visible.npz')['vis'].shape == (n, 2, len(kps))
out = ROOT / 'Review'
out.mkdir(exist_ok=True)
font = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 21)
categories = pose_data['categories'] if 'categories' in pose_data.files else np.array(['start_pose'] + ['mild_grounded'] * 8 + ['medium'] * 8 + ['wide'] * 7)
preview_poses = np.linspace(0, n - 1, min(24, n), dtype=int)
lookup = {(it['pose'], it['view']): i for i, it in enumerate(items)}
main_names = ['pelvis', 'spine_01', 'spine_02', 'spine_03', 'neck_01', 'head'] + [n + s for s in ('l', 'r') for n in ('clavicle_', 'upperarm_', 'lowerarm_', 'hand_', 'thigh_', 'calf_', 'foot_', 'ball_')]
by_label = {str(k[0]): j for j, k in enumerate(kps)}
for pose in preview_poses:
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
        category = {'start_pose': '起始站姿', 'mild_grounded': '温和 · 接地', 'medium': '中等幅度', 'wide': '大幅 · 动态'}[str(categories[pose])]
        draw.text((15, 8), f'{pose + 1:02d}  {category} · 正面 / 侧面', font=font, fill='#202020')
    pair.save(out / f'pose-{pose + 1:02d}.png')
    annotation.save(out / f'pose-{pose + 1:02d}-joints.png')

# 每页 6 对，保留姿态编号；整页所有图都来自实际训练读取结果。
for page in range(4):
    sheet = Image.new('RGB', (1808, 1552), '#f5f5f5')
    d = ImageDraw.Draw(sheet)
    d.text((16, 10), f'训练输入审阅 {page + 1}/4  |  左正面、右侧面  |  原始读取为每视角 448×448', font=font, fill='#202020')
    for tile in range(6):
        im = Image.open(out / f'pose-{preview_poses[page * 6 + tile] + 1:02d}.png')
        sheet.paste(im, (8 + tile % 2 * 900, 52 + tile // 2 * 500))
    sheet.save(out / f'contact-{page + 1:02d}.png')
record = {'stage': 'awaiting_user_review' if n == 24 else 'training_images_ready', 'poses': n, 'views': 2, 'loaded_tensor_shape': list(pictures.shape), 'preview_pose_numbers': (preview_poses + 1).tolist(), 'min_mesh_frame_margin': min_frame_margin, 'raw_format': 'JPEG RGB quality=100', 'render_fit': 896, 'network_size': 448, 'normalisation': 'original reader: BGR to RGB then ImageNet mean/std', 'labels': 'reader/renders/labels.npz', 'visibility': 'reader/visible.npz', 'keypoints': len(kps), 'labels_max_head_error_armature_units': head_error, 'no_training_started': not (ROOT / 'reader' / 'reader.pt').exists(), 'preview_note': 'PNG 为实际加载后的训练输入及独立标注预览；标注图不参与训练。左右视角按同一 pose 分组，完整训练时也应按 pose 划分训练/验证，不能按图片随机拆分。'}
record['amplitude_counts'] = {str(c): int(np.sum(categories == c)) for c in np.unique(categories)}
if 'hand_pose_names' in pose_data.files:
    record['hand_pose_counts'] = {side: {str(name): int(np.sum(pose_data['hand_pose_names'][:, i] == name)) for name in np.unique(pose_data['hand_pose_names'][:, i])} for i, side in enumerate(('left', 'right'))}
(ROOT / 'review-record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, ensure_ascii=False), flush=True)
