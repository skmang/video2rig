"""关键姿势、骨轴转换及动作包共用的数据操作。"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def identifier(value):
    if (not re.fullmatch(r'[A-Z][A-Za-z0-9]{0,63}', value)
            or re.fullmatch(r'CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9]', value, re.IGNORECASE)):
        raise ValueError('ID 使用 PascalCase 字母数字，例如 GuardLeft 或 RightPunch01')
    return value


def library(path):
    root = Path(path).resolve()
    c = read(root / 'character.json')
    expected = digest({k: v for k, v in c.items() if k != 'fingerprint'})
    if c['fingerprint'] != expected:
        raise ValueError('角色基准已改变；请新建基准，不能混用旧姿势和相机')
    return root, c


def pose(root, c, name):
    p = read(root / 'poses' / identifier(name) / 'pose.json')
    if p['character'] != c['fingerprint']:
        raise ValueError('姿势与角色基准不配套')
    if p['fingerprint'] != digest({k: v for k, v in p.items() if k != 'fingerprint'}):
        raise ValueError('姿势内容已变化；使用新 ID 重新捕获和渲染')
    return p


def processing_basis(c, sample):
    rest = {b['name']: np.array(b['rest']) for b in c['rig']['bones']}
    posed = {n: np.array(sample['deform'][n]) @ r for n, r in rest.items()}
    result = {}
    for b in c['rig']['bones']:
        n, parent = b['name'], b['parent']
        if parent:
            local = np.linalg.solve(rest[parent], rest[n])
            result[n] = np.linalg.solve(local, np.linalg.solve(posed[parent], posed[n]))
        else:
            result[n] = np.linalg.solve(rest[n], posed[n])
    return {n: m.tolist() for n, m in result.items()}


def motion_matrices(rig, motion):
    """原版 NPZ 的局部四元数还原为骨架空间矩阵，未拟合骨骼沿用起始姿势。"""
    from scipy.spatial.transform import Rotation
    bones = rig['bones']
    names = [b['name'] for b in bones]
    moved = [str(x) for x in motion['bones']]
    if not set(moved) <= set(names):
        raise ValueError('动画包含当前角色没有的骨骼')
    q = motion['quaternions']
    frames = len(motion['times'])
    if q.shape != (frames, len(moved), 4) or not np.isfinite(q).all():
        raise ValueError('动画四元数形状或数值无效')
    basis = np.array([rig['start_pose'].get(n, np.eye(4).tolist()) for n in names])
    basis = np.broadcast_to(basis, (frames, *basis.shape)).copy()
    for j, n in enumerate(moved):
        basis[:, names.index(n), :3, :3] = Rotation.from_quat(q[:, j, [1, 2, 3, 0]]).as_matrix()
    root = names.index(str(motion['root']))
    basis[:, root, :3, 3] = motion['root_location']
    posed = np.empty_like(basis)
    for j, b in enumerate(bones):
        rest = np.array(b['rest'])
        if b['parent']:
            p = names.index(b['parent'])
            if p >= j:
                raise ValueError('骨骼必须按父级在前排列')
            local = np.linalg.solve(np.array(bones[p]['rest']), rest)
            posed[:, j] = posed[:, p] @ local @ basis[:, j]
        else:
            posed[:, j] = rest @ basis[:, j]
    return names, posed
