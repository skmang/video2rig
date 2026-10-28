"""关键姿势动作的标定和拟合适配；不修改原版源码，不自动启动识别或生成。"""
import copy
import json
from pathlib import Path
import sys

import numpy as np

from data import digest, file_hash, library, pose, read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def unit_scale(rig):
    linear = np.asarray(rig['armature_world'], float)[:3, :3]
    scale = float(np.linalg.norm(linear, axis=0).mean())
    if (not np.isfinite(scale) or scale <= 0 or np.linalg.det(linear) <= 0
            or not np.allclose(linear.T @ linear, np.eye(3) * scale ** 2, rtol=1e-5, atol=1e-12)):
        raise ValueError('只支持正向均匀缩放的骨架；不能用平均缩放处理剪切或镜像')
    return scale


def metric_rig(source):
    """统一骨架空间长度为米，同时保持世界坐标不变；原数据不变。"""
    if source.get('props'):
        raise ValueError('当前关键姿势回传不支持独立道具轨道')
    data = copy.deepcopy(source)
    scale = unit_scale(data)
    for bone in data['bones']:
        rest = np.asarray(bone['rest'], float)
        rest[:3, 3] *= scale
        bone['rest'], bone['length'] = rest.tolist(), bone['length'] * scale
    for name, value in data['start_pose'].items():
        basis = np.asarray(value, float)
        basis[:3, 3] *= scale
        data['start_pose'][name] = basis.tolist()
    world = np.asarray(data['armature_world'], float)
    world[:3, :3] /= scale
    data['armature_world'] = world.tolist()
    return data, scale


def calibrate(a):
    """只读取小型骨架数据，保存相对中性骨架的固定铰链定义。"""
    from scipy.optimize import minimize_scalar
    from video2rig.rig import Rig
    root, character = library(a.library)
    out = root / 'fitting.json'
    if out.exists():
        raise ValueError('标定已存在；修改标定前检查其依赖，不覆盖')
    source = (root / character['project']).resolve()
    data = read(source / 'rig' / 'rig.json')
    if file_hash(source / 'rig' / 'rig.json') != character['source_rig_sha256']:
        raise ValueError('训练基准骨架与角色库不一致')
    cfg = read(source / 'creature.json')
    ways = cfg.get('hinges')
    if not isinstance(ways, dict) or not ways:
        raise ValueError('须先明确需标定的肘膝骨名及 front/back 方向')
    rig = Rig(data, [b['name'] for b in data['bones']], cfg['root_bone'])
    p0 = rig.start_params()
    directions = {'+X': (1, 0, 0), '-X': (-1, 0, 0), '+Y': (0, 1, 0), '-Y': (0, -1, 0)}
    facing = np.linalg.solve(np.asarray(data['armature_world'])[:3, :3], directions[cfg['facing']])
    facing /= np.linalg.norm(facing)
    heads, tails = rig.bone_points(p0)
    hinges = {}
    for name, way in ways.items():
        if way not in ('front', 'back'):
            raise ValueError(f'{name}: 不支持的铰链方向 {way}')
        j = rig.names.index(name)
        parent = rig.parents[j]
        if parent < 0:
            raise ValueError('根骨不能作为肘膝铰链')
        # 在固定基准中测量哪条横轴产生朝向前后的弯曲，不依赖动作护卫的肩部旋转。
        delta = {}
        for axis in (0, 2):
            q = p0.copy()
            q[rig.index(name, axis)] += .05
            delta[axis] = float((rig.bone_points(q)[1][j] - tails[j]) @ facing)
        axis = max(delta, key=lambda ax: abs(delta[ax]))
        if abs(delta[axis]) < rig.length[j] * 1e-4:
            raise ValueError(f'{name}: 基准不能可靠判断弯曲方向，需要三维标定')
        sign = (1 if delta[axis] > 0 else -1) * (1 if way == 'front' else -1)
        channel = rig.index(name, axis)

        def off(value):
            q = p0.copy()
            q[channel] = value
            h, t = rig.bone_points(q)
            u, v = t[parent] - h[parent], t[j] - h[j]
            return -float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

        straight = float(minimize_scalar(off, bounds=(-np.pi, np.pi), method='bounded').x)
        # 当前人形生产的保守初值；最终姿势与动态仍须审阅，不宣称自动解剖验收。
        flex = np.radians(150 if way == 'front' else 140)
        bounds = sorted([straight - sign * np.radians(2), straight + sign * flex])
        hinges[name] = {'axis': axis, 'sign': sign, 'straight': straight, 'range': bounds}
    record = {'version': 1, 'character': character['fingerprint'],
              'source_rig_sha256': character['source_rig_sha256'],
              'metres_per_rig_unit': unit_scale(data), 'hinges': hinges,
              'fps': 24, 'camera': 'fixed', 'floor_world_z': character['floor_z'],
              'ignored_keypoint_prefixes': ['ik_'],
              'scope': '静态接口标定；角度范围为当前人形初值，尚未完成动态验收'}
    record['fingerprint'] = digest(record)
    write(out, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def job_contract(job):
    job = Path(job).resolve()
    prepared = read(job / 'prepared.json')
    root, character = library(job / prepared['library'])
    calibration = read(root / 'fitting.json')
    if calibration['fingerprint'] != digest({k: v for k, v in calibration.items() if k != 'fingerprint'}):
        raise ValueError('拟合标定内容与指纹不一致')
    if not prepared['character'] == calibration['character'] == character['fingerprint']:
        raise ValueError('动作与拟合标定不属于同一角色')
    action = read(job / 'action.json')
    for field in ('start', 'end'):
        endpoint = pose(root, character, action[field])
        if endpoint['fingerprint'] != prepared[field]:
            raise ValueError('动作端点与共享姿势不一致')
    rig = read(job / 'rig' / 'rig.json')
    start = pose(root, character, action['start'])
    if (rig['start_pose'] != start['processing_basis']
            or rig['armature_world'] != start['sample']['armature_world']
            or rig['bones'] != character['rig']['bones']):
        raise ValueError('动作骨架的起点、物体矩阵或静止矩阵不匹配')
    if read(job / 'start' / 'cameras.json') != character['cameras']:
        raise ValueError('动作相机与角色标定不匹配')
    if file_hash(job / 'rig' / 'mesh.npz') != character['rig_assets']['mesh.npz']:
        raise ValueError('动作网格与角色库基准不匹配')
    if not np.isclose(unit_scale(rig), calibration['metres_per_rig_unit'], rtol=1e-6):
        raise ValueError('动作骨架单位与标定不匹配')
    if action['root_motion'] != 'in_place':
        raise ValueError('当前适配先覆盖原地直拳/格挡；根运动须单独定义接触与行进合同')
    return root, character, calibration, action


def calibrated_creature(project, calibration):
    """原版 Creature 的实例级扩展，不替换第三方类或全局函数。"""
    from video2rig.creature import Creature
    from video2rig.rig import Rig

    class FixedCreature(Creature):
        def p0(self, clip):
            return self.rig.start_params()

        def bend_axis(self, j):
            rec = calibration['hinges'].get(self.rig.names[j])
            return rec['axis'] if rec else super().bend_axis(j)

        def find_hinges(self):
            self.reversed = set()
            return {n: r['sign'] for n, r in calibration['hinges'].items()}

        def bounds(self, clip):
            lo, hi, p0 = super().bounds(clip)
            for name, rec in calibration['hinges'].items():
                k = self.rig.index(name, rec['axis'])
                lo[k], hi[k] = rec['range']
            if np.any(p0 < lo) or np.any(p0 > hi):
                raise ValueError('动作起点超出固定关节范围；不能通过裁剪起点隐藏错误')
            return lo, hi, p0

        def views(self, clip):
            return self.fixed_views

        def times(self, clip):
            return self.fixed_times.copy()

    cr = FixedCreature(project, project.clip_cfg(''))
    names = [cr.rig.names[j] for j in cr.rig.chan_of]
    cr.source_data = cr.data
    cr.data, cr.unit = metric_rig(cr.data)
    cr.mesh = {**cr.mesh, 'coords': cr.mesh['coords'] * cr.unit}
    cr.rig = Rig(cr.data, names, project.cfg['root_bone'])
    cr.n = cr.rig.n
    cr.height *= cr.unit
    cr.floor = calibration['floor_world_z'] - np.asarray(cr.data['armature_world'])[2, 3]
    cr.role = cr.axis_roles()
    cr.legs = cr.standing_legs()
    cr.bounds(None)
    return cr


def fit(a):
    """显式执行后才读视频、求解；默认仅检查输入并显示任务。"""
    root, character, calibration, action = job_contract(a.job)
    if not a.execute:
        print(f'待执行拟合：{action["id"]}；固定相机、24fps、米制求解、固定肘膝；未启动。')
        return
    from threadpoolctl import threadpool_limits
    from video2rig import clip as clipmod
    from video2rig.common import Project
    from video2rig.reader import KeypointRig
    from video2rig.reader.fit import SETTINGS, fix_twins, panels_agree, fit_hidden
    from motion import duration
    project = Project(a.job)
    name = action['id']
    raw_path = project.root / 'work' / name / 'reader-keyframes.npz'
    provenance = read(raw_path.with_suffix('.json'))
    expected = reading_identity(project.root, root)
    if provenance['inputs'] != expected or provenance['npz_sha256'] != file_hash(raw_path):
        raise ValueError('识别结果与当前视频、相机、骨架、reader 或标定不配套，须重新识别')
    out = project.root / 'out' / f'{name}.npz'
    evidence = out.with_suffix('.fit.json')
    used = raw_path.with_name('observations-used.npz')
    if any(p.exists() for p in (out, evidence, used)):
        raise ValueError('拟合产物已存在，不覆盖；先检查当前候选和保留关系')
    with np.load(raw_path, allow_pickle=False) as archive:
        uv, conf = archive['uv'].copy(), archive['conf'].copy()
        kps = json.loads(str(archive['keypoints']))
        views = archive['views'].tolist()
    frames = clipmod.read_frames(project.clip_file(name))
    params = read(project.root / 'parameters.json')
    count = duration(params['category'], params['seconds'])
    if len(frames) != count or uv.shape != (count, 2, len(kps), 2) or conf.shape != uv.shape[:-1]:
        raise ValueError('视频帧数或关节数据形状与动作合同不一致')
    if not np.isfinite(uv).all() or not np.isfinite(conf).all():
        raise ValueError('识别结果含非有限数值')
    if any(f.shape[:2] != (params['height'], params['width']) for f in frames):
        raise ValueError('视频实际尺寸与动作合同不一致')
    cr = calibrated_creature(project, calibration)
    cams = character['cameras']
    if views != [r['name'] for r in cams['views']]:
        raise ValueError('识别视角顺序与相机不一致')
    pivot = cr.rig.bone_points(cr.p0(None))[0][cr.rig.root]
    cr.fixed_views = []
    for rec in cams['views']:
        cam = clipmod.Camera(rec, (params['width'], params['height']), cams['frame_px'],
                             cr.data['armature_world'], pivot)
        cr.fixed_views.append(dict(name=rec['name'], size=cam.size, project=cam.project,
                                   depth=cam.depth, cam=cam, panel=cam.panel,
                                   frame=lambda i, pn=cam.panel: pn.cut(frames[i]).copy()))
    cr.fixed_times = np.arange(count, dtype=float) / calibration['fps']
    metric_kps = [(label, bone, (np.asarray(f) * cr.unit).tolist() if not np.isscalar(f) else f)
                  for label, bone, f in kps]
    ignored = [i for i, k in enumerate(kps) if any(k[0].startswith(p) for p in calibration['ignored_keypoint_prefixes'])]
    conf[:, :, ignored] = 0
    settings = {**SETTINGS, 'steps': False, 'panel_drift': False}
    with threadpool_limits(limits=1):
        use_uv, swaps = fix_twins(cr, name, uv.copy(), conf, metric_kps, s=settings)
        use_uv, use_conf = panels_agree(cr, name, use_uv, conf.copy(), metric_kps, s=settings)
        use_conf[:, :, ignored] = 0
        motion, report = fit_hidden(cr, name, use_uv, use_conf, metric_kps, s=settings)
    # 导出回原处理骨架单位，才能继续使用已有原骨架回传和 join；不修改首尾帧。
    quats, locations = [], []
    for p in motion:
        q, loc, _ = cr.rig.pose_basis(p)
        quats.append(q)
        locations.append(loc / cr.unit)
    kr = KeypointRig(cr, metric_kps)
    projected = np.array([[v['project'](kr.points(p)) for v in cr.fixed_views] for p in motion])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, bones=np.array(cr.rig.names), root=cr.rig.names[cr.rig.root],
                        quaternions=np.asarray(quats), root_location=np.asarray(locations),
                        props=np.array([], dtype=str), prop_matrices=np.empty((count, 0, 4, 4)),
                        times=cr.fixed_times, armature_world=np.asarray(cr.source_data['armature_world']))
    np.savez_compressed(used, uv=use_uv, conf=use_conf, projected=projected,
                        raw_error_px=np.linalg.norm(projected - uv, axis=-1),
                        used_error_px=np.linalg.norm(projected - use_uv, axis=-1),
                        keypoints=json.dumps(kps), views=np.array(views), swaps=json.dumps(swaps))
    write(evidence, {'inputs': expected, 'motion_sha256': file_hash(out), 'settings': settings,
                     'ignored_keypoints': [kps[i][0] for i in ignored], 'frames': count,
                     'status': 'fitted_not_accepted', 'report': report,
                     'remaining': ['原地位移及鞋底接触验收', '首尾姿势和速度修正', '手型与蒙皮动态审阅']})
    print(f'拟合已保存 {out}；尚未完成动作验收')


def reading_identity(job, root):
    job, root = Path(job), Path(root)
    action = read(job / 'action.json')
    paths = {'video': job / 'clips' / f'{action["id"]}.mp4',
             'reader': (root / read(root / 'character.json')['project']) / 'reader' / 'reader.pt',
             'camera': job / 'start' / 'cameras.json', 'rig': job / 'rig' / 'rig.json',
             'mesh': job / 'rig' / 'mesh.npz',
             'calibration': root / 'fitting.json', 'parameters': job / 'parameters.json',
             'creature': job / 'creature.json', 'action': job / 'action.json',
             'adapter': Path(__file__), 'reader_source': Path(__file__).resolve().parents[2] / 'video2rig' / 'reader' / 'read.py',
             'fit_source': Path(__file__).resolve().parents[2] / 'video2rig' / 'reader' / 'fit.py'}
    return {name: file_hash(path) for name, path in paths.items()}


def recognise(a):
    root, _, _, action = job_contract(a.job)
    from training import offline_environment
    env = offline_environment(a.hf_home)
    if not a.execute:
        print(f'待执行识别：{action["id"]}；共享新版 reader，8 次平移缩放采样；未启动。')
        return
    import os
    os.environ.update({k: env[k] for k in ('HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')})
    from video2rig.common import Project
    from video2rig.reader import keypoints
    from video2rig.creature import Creature
    from video2rig.reader.road import read_clip
    from video2rig.reader.read import panels, sheet
    from video2rig import clip as clipmod
    project = Project(a.job)
    name = action['id']
    out = project.root / 'work' / name / 'reader-keyframes.npz'
    film = out.with_name('reader-keyframes-overlay.mp4')
    if any(p.exists() for p in (out, out.with_suffix('.json'), film, out.with_name('reader-keyframes-sheet.jpg'))):
        raise ValueError('识别结果已存在，不覆盖')
    identity = reading_identity(project.root, root)
    reader = (root / read(root / 'character.json')['project']) / 'reader' / 'reader.pt'
    # 先核对 checkpoint 标签，再运行推理，避免 86/80 标签或同名不同坐标混用。
    import torch
    checkpoint = torch.load(reader, map_location='cpu', weights_only=False)
    cr = Creature(project, project.clip_cfg(name))
    if digest(checkpoint['keypoints']) != digest(keypoints(cr)):
        raise ValueError('reader 关节语义与当前角色不匹配')
    del checkpoint, cr
    uv, conf, kps, names = read_clip(project, name, reader=reader, tag='-keyframes', cache=False, tta=8, crops=False)
    frames = clipmod.read_frames(project.clip_file(name))
    cams = read(project.start_dir / 'cameras.json')
    if len(frames) != len(uv) or not np.isfinite(uv).all() or not np.isfinite(conf).all():
        raise ValueError('识别结果与视频帧数不符或含非有限数值')
    picks = np.unique(np.linspace(0, len(frames) - 1, min(12, len(frames))).astype(int)).tolist()
    sheet(frames, panels(frames, cams), uv, conf, [k[0] for k in kps], picks,
          out.with_name('reader-keyframes-sheet.jpg'))
    overlay(frames, panels(frames, cams), uv, conf, [k[0] for k in kps], film)
    write(out.with_suffix('.json'), {'inputs': identity, 'npz_sha256': file_hash(out),
                                    'status': 'read_not_accepted', 'preview_frames': picks,
                                    'overlay': film.name, 'overlay_fps': 24, 'overlay_frames': len(frames),
                                    'preview_scope': '均匀抽帧；另须检查动作极值、遮挡及全段身份连续性'})


def overlay(frames, views, uv, conf, labels, path):
    """真实视频全部帧的关节点叠加，不插帧；低峰值空心、左右分色。"""
    import cv2
    from video2rig.reader.read import colour
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 24, (w, h))
    if not writer.isOpened():
        raise ValueError('当前 OpenCV 无法打开 MP4 编码器')
    try:
        for i, frame in enumerate(frames):
            result = frame.copy()
            for v, pv in enumerate(views):
                panel = pv['panel']
                img = panel.cut(frame).copy()
                for k, (label, point) in enumerate(zip(labels, panel.from_camera(uv[i, v]))):
                    if label.startswith('ik_'):
                        continue
                    x, y = np.rint(point).astype(int)
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        cv2.circle(img, (x, y), 3, colour(label), -1 if conf[i, v, k] >= .3 else 1,
                                   lineType=cv2.LINE_AA)
                panel.put(result, img)
            cv2.putText(result, f'frame {i} / {len(frames) - 1}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(result)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, decoded = cap.read()
            if not ok:
                break
            if decoded.shape[:2] != (h, w):
                raise ValueError('叠加视频回读尺寸不一致')
            count += 1
    finally:
        cap.release()
    if count != len(frames):
        raise ValueError('叠加视频回读帧数不一致')
