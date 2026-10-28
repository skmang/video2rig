"""共享 reader 的显式训练及分关节评估入口；不生成训练图片。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from data import digest, file_hash, library, read, write


def offline_environment(home):
    home = Path(home).resolve()
    snapshots = home / 'hub' / 'models--facebook--dinov2-small' / 'snapshots'
    if not any((p / 'config.json').is_file() and (p / 'model.safetensors').is_file()
               for p in snapshots.glob('*')):
        raise ValueError('指定位置没有完整本地 DINO 缓存；本命令不会下载')
    return {**os.environ, 'HF_HOME': str(home), 'HF_HUB_CACHE': str(home / 'hub'),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}


def dataset_identity(project, renders):
    from video2rig.reader.train import render_file
    directory = project.root / renders
    repo = Path(__file__).resolve().parents[2]
    files = {'rig': project.rig_dir / 'rig.json', 'mesh': project.rig_dir / 'mesh.npz',
             'creature': project.root / 'creature.json',
             'train_source': repo / 'video2rig' / 'reader' / 'train.py',
             'network_source': repo / 'video2rig' / 'reader' / 'net.py',
             'cameras': project.start_dir / 'cameras.json', 'labels': directory / 'labels.npz',
             'poses': render_file(directory, 'poses.npz'), 'visible': render_file(directory, 'visible.npz')}
    return {name: file_hash(path) for name, path in files.items()}


def verify_cache(cr, directory):
    """原版 params.npy 只按形状命中；用实际姿态小样本检查，不写缓存。"""
    from video2rig.reader import render_params
    from video2rig.reader.train import render_file
    path = directory / 'params.npy'
    if not path.exists():
        return
    with np.load(directory / 'labels.npz', allow_pickle=False) as lab, np.load(
            render_file(directory, 'poses.npz'), allow_pickle=False) as poses:
        count = len(lab['index'])
        selected = np.unique(np.linspace(0, count - 1, min(count, 16)).astype(int))
        subset = {k: lab[k] for k in lab.files}
        for key in ('index', 'heads', 'tails'):
            subset[key] = subset[key][selected]
        actual = render_params(cr, poses, subset)
        cached = np.load(path, allow_pickle=False)
        if cached.shape != (count, cr.rig.n) or not np.allclose(cached[selected], actual, atol=1e-6):
            raise ValueError('params.npy 与当前姿态不一致；检查并清理该数据集旧缓存后重建')


def train(a):
    root, character = library(a.library)
    project_path = (root / character['project']).resolve()
    if a.steps <= 0 or a.batch <= 0:
        raise ValueError('steps 和 batch 必须为正数')
    env = offline_environment(a.hf_home)
    command = [sys.executable, '-B', '-m', 'video2rig.reader.train', str(project_path),
               '--steps', str(a.steps), '--batch', str(a.batch), '--seed', '20261028']
    if not a.execute:
        print(json.dumps({'task': 'train', 'command': command, 'offline': True,
                          'status': 'waiting_for_explicit_start'}, ensure_ascii=False, indent=2))
        return
    os.environ.update({k: env[k] for k in ('HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')})
    from video2rig.common import Project
    from video2rig.creature import Creature
    from video2rig.reader import keypoints
    project = Project(project_path)
    if file_hash(project.rig_dir / 'rig.json') != character['source_rig_sha256']:
        raise ValueError('当前训练骨架与角色库基准不一致')
    out = project.root / 'reader' / 'reader.pt'
    provenance = out.with_suffix('.inputs.json')
    if any(p.exists() for p in (out, provenance, out.with_name('reader-train.json'))):
        raise ValueError('共享 reader 或训练记录已存在，不覆盖')
    cr = Creature(project, project.clip_cfg(''))
    verify_cache(cr, project.root / 'reader' / 'renders')
    inputs = dataset_identity(project, 'reader/renders')
    labels = keypoints(cr)
    del cr
    subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env, check=True)
    write(provenance, {'inputs': inputs, 'keypoints': labels, 'reader_sha256': file_hash(out),
                       'command': command, 'status': 'trained_not_accepted'})
    print('训练结束；尚须独立姿态及真实视频识别验收，不自动开始生成或拟合')


def evaluate(a):
    root, character = library(a.library)
    project_path = (root / character['project']).resolve()
    env = offline_environment(a.hf_home)
    if not 0 < a.confidence <= 1:
        raise ValueError('confidence 应在 (0, 1]；它仅为热图峰值分组阈值')
    if not a.execute:
        print(json.dumps({'task': 'evaluate', 'project': str(project_path), 'renders': a.renders,
                          'selection': 'independent_all' if a.all_poses else 'heldout_5_percent',
                          'status': 'waiting_for_explicit_start'}, ensure_ascii=False, indent=2))
        return
    os.environ.update({k: env[k] for k in ('HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')})
    import torch
    from video2rig.common import Project
    from video2rig.creature import Creature
    from video2rig.reader import keypoints
    from video2rig.reader.read import load_reader
    from video2rig.reader.net import decode, normalise, letterbox
    from video2rig.reader.train import load_items, load_pictures, HOLD_OUT
    project = Project(project_path)
    directory = (project.root / a.renders).resolve()
    if not directory.is_relative_to(project.root):
        raise ValueError('评估输入须位于训练项目内')
    if a.all_poses and directory == (project.root / 'reader' / 'renders').resolve():
        raise ValueError('训练集不能作为全量独立评估集')
    out = directory / ('reader-independent-quality.json' if a.all_poses else 'reader-heldout-quality.json')
    if out.exists():
        raise ValueError('评估报告已存在，不覆盖')
    cr = Creature(project, project.clip_cfg(''))
    labels = keypoints(cr)
    verify_cache(cr, directory)
    inputs = dataset_identity(project, a.renders)
    items = load_items(project, labels, directory, cr=cr)
    pose_ids = np.array([it['pose'] for it in items])
    cut = np.quantile(pose_ids, 1 - HOLD_OUT)
    items = [it for it in items if a.all_poses or it['pose'] > cut]
    if not items:
        raise ValueError('没有独立评估样本')
    imgs, truth, _ = load_pictures(items)
    device = torch.device('cuda')
    path = project.root / 'reader' / 'reader.pt'
    model, ck = load_reader(path, device)
    if digest(ck['keypoints']) != digest(labels) or model.context or model.depth:
        raise ValueError('本入口评估当前无 context/depth 的同标签 reader')
    predictions, confidence = [], []
    with torch.inference_mode():
        for i in range(0, len(imgs), 24):
            x = normalise(torch.from_numpy(imgs[i:i + 24]).to(device))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                heat = model(x)
            p, c = decode(heat.float())
            predictions.append(p.cpu().numpy())
            confidence.append(c.cpu().numpy())
    p, confidence = np.concatenate(predictions), np.concatenate(confidence)
    scale = np.array([letterbox(it['size'])[0] * it['per_clip_px'] for it in items])
    error = np.linalg.norm(p - truth, axis=-1) / scale[:, None]
    visible = np.stack([it['seen'] & it['inside'] for it in items])
    inside = np.stack([it['inside'] for it in items])
    view_ids = np.array([it['view'] for it in items])
    cameras = read(project.start_dir / 'cameras.json')['views']
    s = float(np.linalg.norm(np.asarray(cr.data['armature_world'])[:3, 2]))
    reports = {}
    for v, camera in enumerate(cameras):
        sel = view_ids == v
        height_px = cr.height * s / camera['ortho_height'] * camera['height_px']
        reports[camera['name']] = {}
        for k, kp in enumerate(labels):
            if kp[0].startswith('ik_'):
                continue
            seen = sel & visible[:, k]
            hidden = sel & inside[:, k] & ~visible[:, k]
            sure = seen & (confidence[:, k] >= a.confidence)
            e = error[seen, k]
            reports[camera['name']][kp[0]] = {
                'visible_count': int(seen.sum()), 'hidden_count': int(hidden.sum()),
                'median_px': float(np.median(e)) if len(e) else None,
                'p95_px': float(np.percentile(e, 95)) if len(e) else None,
                'p95_height_fraction': float(np.percentile(e, 95) / height_px) if len(e) else None,
                'confident_count': int(sure.sum()),
                'confident_error_over_2pct_fraction': float((error[sure, k] > .02 * height_px).mean()) if sure.any() else None,
                'hidden_high_peak_fraction': float((confidence[hidden, k] >= a.confidence).mean()) if hidden.any() else None}
    write(out, {'inputs': inputs, 'reader_sha256': file_hash(path), 'pictures': len(items),
                'poses': len(set(it['pose'] for it in items)), 'confidence_peak_threshold': a.confidence,
                'pixel_basis': '标定 cameras.json 的分栏像素；不是生成视频像素',
                'scope': '独立全量输入' if a.all_poses else '按姿态编号留出的末尾5%',
                'status': 'measured_not_accepted', 'views': reports,
                'remaining': '真实视频的左右身份、事件同步及遮挡段仍须审阅'})
    print(f'已保存分关节评估：{out}；未自动判定识别合格')
