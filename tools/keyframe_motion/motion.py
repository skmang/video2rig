"""准备 H3 首尾关键帧动作包；只写文件，不调用生成接口。"""
import argparse
import copy
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

from data import digest, file_hash, identifier, library, motion_matrices, pose, read, write

REPO = Path(__file__).resolve().parents[2]


def duration(category, seconds):
    low, high = {'short': (1, 3), 'long': (3, 5)}[category]
    if not low <= seconds <= high:
        raise ValueError(f'{category} 时长必须在 {low}～{high} 秒')
    choices = [17 * k + 5 for k in range(20) if low <= (17 * k + 5) / 24 <= high]
    return min(choices, key=lambda n: (abs(n / 24 - seconds), n))


def prepare(a):
    root, c = library(a.library)
    contract = read(a.contract)
    allowed = {'id', 'start', 'end', 'category', 'seconds', 'root_motion', 'motion', 'rhythm'}
    if set(contract) != allowed:
        raise ValueError(f'动作合同字段不匹配：缺少 {allowed - set(contract)}，多余 {set(contract) - allowed}；本入口只支持首尾图')
    name = identifier(contract['id'])
    if contract['root_motion'] not in ('in_place', 'root_motion'):
        raise ValueError('root_motion 必须是 in_place 或 root_motion')
    for field in ('motion', 'rhythm'):
        if not contract[field].strip():
            raise ValueError(f'缺少动作合同字段 {field}')
    frames = duration(contract['category'], contract['seconds'])
    endpoints = [pose(root, c, contract[k]) for k in ('start', 'end')]
    for p in endpoints:
        if p['purpose'] != 'action':
            raise ValueError('标定姿势不能作为动作端点')
        directory = root / 'poses' / p['id']
        r = read(directory / 'render.json')
        if r['pose'] != p['fingerprint'] or r['camera'] != digest(c['cameras']):
            raise ValueError('关键图与姿势/相机不配套，请重新渲染')
        if r['image_sha256'] != file_hash(directory / 'frame.png'):
            raise ValueError('关键图被修改，禁止混用旧标定')
        if not r['inside_frame']:
            raise ValueError('关键姿势有网格出框')
    # 只生成新包，不覆盖已有任务或已生成视频。
    out = root / 'jobs' / name
    if out.exists():
        raise ValueError(f'动作包已存在，请使用新 ID：{out}')
    width, height = c['video_size']
    if width % 32 or height % 32 or width * c['cameras']['frame_px'][1] != height * c['cameras']['frame_px'][0]:
        raise ValueError('输出须为 32 的倍数且保持标定画幅比例')
    source = (root / c['project']).resolve()
    for filename, expected in c['rig_assets'].items():
        if file_hash(source / 'rig' / filename) != expected:
            raise ValueError(f'基准骨架资源已变化：{filename}')
    rig = copy.deepcopy(c['rig'])
    rig['start_pose'] = endpoints[0]['processing_basis']
    rig['armature_world'] = endpoints[0]['sample']['armature_world']
    # 原版只支持 root 的平移；其它骨骼的位移或缩放不可无声丢弃。
    root_bone = read(source / 'creature.json').get('root_bone', 'pelvis')
    for p in endpoints:
        for bone, value in p['processing_basis'].items():
            m = np.array(value)
            if not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-4):
                raise ValueError(f'{bone} 存在非刚性缩放，原版拟合不支持')
            if bone != root_bone and not np.allclose(m[:3, 3], 0, atol=1e-3):
                raise ValueError(f'{bone} 存在非根骨平移，原版拟合不支持')
    if contract['root_motion'] == 'in_place':
        if not np.allclose(endpoints[0]['sample']['armature_world'], endpoints[1]['sample']['armature_world'], atol=1e-5):
            raise ValueError('原地动作的首尾物体变换不同')
    prompt = (
        f'参考图片与目标视频的对应关系：Picture 1 对应 Shot 1 的首帧；Picture 2 对应同一镜头的尾帧。'
        f'实际片长 {frames / 24:.3f} 秒，尾帧位于 {(frames - 1) / 24:.3f} 秒。\n\n'
        f'integrated_multimodal_description: [Shot 1] {contract["motion"].strip()}\n\n'
        f'动作节奏：{contract["rhythm"].strip()}\n\n'
        '从 Picture 1 的动作姿势开始，以 Picture 2 的动作姿势结束，中间身体协调自然，动作连续有力度。'
        '左栏正面、右栏角色左侧面，同步展示同一角色的同一次动作。两个镜头固定，保持原有构图与人物尺度，'
        '全身与伸展中的手脚完整入镜，运动轮廓清楚，沿用参考角色外观，背景和照明稳定。\n\n'
        'overall_soundscape: 自然的发力呼吸、脚步和衣料摩擦声。\n\nnon_diegetic_music: N/A\n'
    )
    template = REPO / 'workflows' / 'minimax_h3'
    api = read(template / 'MedievalGirl_ComplexAction.api.json')
    ui = read(template / 'MedievalGirl_ComplexAction.workflow.json')
    image_names = [f'{c["id"]}_{p["id"]}_{p["fingerprint"][:10]}.png' for p in endpoints]
    api['1']['inputs']['image'] = image_names[0]
    api['16'] = {'class_type': 'LoadImage', 'inputs': {'image': image_names[1]}, '_meta': {'title': '共享尾姿势'}}
    api['6']['inputs'].update(prompt=prompt.rstrip(), width=width, height=height, length=frames, last_frame=['16', 0])
    api['6']['_meta']['title'] = f'{name}：首尾关键姿势'
    api['15']['inputs']['filename_prefix'] = f'video2rig/{c["id"]}/{name}'
    nodes = {n['id']: n for n in ui['nodes']}
    end_node = copy.deepcopy(nodes[1])
    end_node.update(id=16, title='共享尾姿势', pos=[0, -420], order=0)
    end_node['widgets_values'][0] = image_names[1]
    end_node['outputs'][0]['links'] = [20]
    nodes[1]['widgets_values'][0] = image_names[0]
    nodes[1]['title'] = '共享首姿势'
    nodes[6]['widgets_values'] = [prompt.rstrip(), width, height, frames]
    nodes[6]['title'] = api['6']['_meta']['title']
    end_index = next(i for i, x in enumerate(nodes[6]['inputs']) if x['name'] == 'last_frame')
    nodes[6]['inputs'][end_index]['link'] = 20
    nodes[15]['widgets_values'][0] = api['15']['inputs']['filename_prefix']
    ui['nodes'].append(end_node)
    ui['links'].append([20, 16, 0, 6, end_index, 'IMAGE'])
    ui.update(id=str(uuid.uuid4()), last_node_id=16, last_link_id=20, revision=0)
    # 如需安装输入，提前检查重名内容，避免覆盖他人素材。
    if a.comfy_input:
        folder = Path(a.comfy_input).resolve()
        if not folder.is_dir():
            raise ValueError('ComfyUI input 目录不存在')
        for p, filename in zip(endpoints, image_names):
            dest = folder / filename
            src = root / 'poses' / p['id'] / 'frame.png'
            if dest.exists() and file_hash(dest) != file_hash(src):
                raise ValueError(f'ComfyUI 输入重名且内容不同：{dest}')
    out.mkdir(parents=True)
    for part in ('rig', 'start', 'comfy-input'):
        (out / part).mkdir()
    for file in (source / 'rig').iterdir():
        if file.is_file() and file.name != 'rig.json':
            shutil.copy2(file, out / 'rig' / file.name)
    write(out / 'rig' / 'rig.json', rig)
    write(out / 'start' / 'cameras.json', c['cameras'])
    for i, (p, filename) in enumerate(zip(endpoints, image_names)):
        src = root / 'poses' / p['id'] / 'frame.png'
        shutil.copy2(src, out / 'start' / ('start.png' if i == 0 else 'end.png'))
        shutil.copy2(src, out / 'comfy-input' / filename)
        if a.comfy_input and not (folder / filename).exists():
            shutil.copy2(src, folder / filename)
    cfg = read(source / 'creature.json')
    for key in ('clips', 'rig', 'start'):
        cfg.pop(key, None)
    cfg['name'] = name
    cfg['clips'] = {name: {'file': f'clips/{name}.mp4'}}
    write(out / 'creature.json', cfg)
    write(out / 'workflow.api.json', api)
    write(out / 'workflow.json', ui)
    (out / 'prompt.txt').write_text(prompt, encoding='utf-8')
    write(out / 'action.json', contract)
    write(out / 'prepared.json', {
        'character': c['fingerprint'], 'library': os.path.relpath(root, out).replace('\\', '/'),
        'start': endpoints[0]['fingerprint'], 'end': endpoints[1]['fingerprint'],
        'frames': frames, 'fps': 24, 'duration': frames / 24, 'last_frame_time': (frames - 1) / 24,
        'reader': 'not_attached: validate compatibility or train for this project before fitting',
        'status': 'prepared_not_generated', 'api_sha256': file_hash(out / 'workflow.api.json'),
        'ui_sha256': file_hash(out / 'workflow.json'),
        'source_rig_sha256': file_hash(source / 'rig' / 'rig.json'),
    })
    print(f'已准备 {out}；{frames} 帧 / {frames / 24:.3f} 秒；未提交生成，未挂接识别器')


def join(a):
    from scipy.spatial.transform import Rotation
    ja, jb = Path(a.first).resolve(), Path(a.second).resolve()
    ma, mb = read(ja / 'prepared.json'), read(jb / 'prepared.json')
    ca, cb = read(ja / 'action.json'), read(jb / 'action.json')
    result = {'same_character': ma['character'] == mb['character'],
              'shared_boundary': ma['end'] == mb['start'],
              'same_root_policy': ca['root_motion'] == cb['root_motion']}
    la, character = library(ja / ma['library'])
    lb, character_b = library(jb / mb['library'])
    pa, pb = pose(la, character, ca['end']), pose(lb, character_b, cb['start'])
    result['same_contacts'] = pa['contacts'] == pb['contacts']
    if bool(a.motion_first) != bool(a.motion_second):
        raise ValueError('检查实际动作时必须同时给出两个 NPZ')
    if a.motion_first:
        arrays = []
        for job, path in ((ja, a.motion_first), (jb, a.motion_second)):
            with np.load(path, allow_pickle=False) as archive:
                m = dict(archive)
            times = m['times']
            if len(times) < 2 or np.any(np.diff(times) <= 0):
                raise ValueError('实际动作须至少两帧且时间递增')
            names, matrices = motion_matrices(read(job / 'rig' / 'rig.json'), m)
            world = np.array(m['armature_world']) @ matrices
            arrays.append((names, times, world))
        na, ta, wa = arrays[0]
        nb, tb, wb = arrays[1]
        if na != nb:
            raise ValueError('两条动作骨骼列表不一致')
        ra, rb = wa[-1, :, :3, :3], wb[0, :, :3, :3]
        ra = ra / np.linalg.norm(ra, axis=1)[:, None, :]
        rb = rb / np.linalg.norm(rb, axis=1)[:, None, :]
        result['max_boundary_angle_deg'] = float(np.degrees(Rotation.from_matrix(np.swapaxes(ra, 1, 2) @ rb).magnitude()).max())
        result['max_boundary_position_m'] = float(np.linalg.norm(wa[-1, :, :3, 3] - wb[0, :, :3, 3], axis=1).max())
        va = (wa[-1, :, :3, 3] - wa[-2, :, :3, 3]) / (ta[-1] - ta[-2])
        vb = (wb[1, :, :3, 3] - wb[0, :, :3, 3]) / (tb[1] - tb[0])
        result['max_boundary_velocity_difference_m_s'] = float(np.linalg.norm(va - vb, axis=1).max())
        result['contact_bones'] = {
            n: {'boundary_position_difference_m': float(np.linalg.norm(wa[-1, na.index(n), :3, 3] - wb[0, na.index(n), :3, 3])),
                'outgoing_speed_m_s': float(np.linalg.norm(va[na.index(n)])),
                'incoming_speed_m_s': float(np.linalg.norm(vb[na.index(n)]))}
            for n in sorted(set(pa['contacts']) | set(pb['contacts']))}
        previous = wa[-2, :, :3, :3]
        previous = previous / np.linalg.norm(previous, axis=1)[:, None, :]
        following = wb[1, :, :3, :3]
        following = following / np.linalg.norm(following, axis=1)[:, None, :]
        angular_a = Rotation.from_matrix(np.swapaxes(previous, 1, 2) @ ra).as_rotvec() / (ta[-1] - ta[-2])
        angular_b = Rotation.from_matrix(np.swapaxes(rb, 1, 2) @ following).as_rotvec() / (tb[1] - tb[0])
        result['max_boundary_angular_velocity_difference_deg_s'] = float(np.degrees(np.linalg.norm(angular_a - angular_b, axis=1)).max())
        result['coverage'] = 'all recorded bone heads and rotations; one-sided adjacent-frame velocities; no root realignment'
    else:
        result['coverage'] = 'boundary contract only; actual motion and velocity not checked'
    if a.out:
        write(a.out, result)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(result[k] for k in ('same_character', 'shared_boundary', 'same_root_policy', 'same_contacts')):
        raise SystemExit(2)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='只生成首尾图配套动作包，不排队')
    p.add_argument('library', type=Path)
    p.add_argument('contract', type=Path)
    p.add_argument('--comfy-input', help='可选：将已渲染输入安装至已有 ComfyUI input 目录')
    p.set_defaults(run=prepare)
    p = commands.add_parser('join', help='检查共享边界及可选的实际 NPZ 边界差异')
    p.add_argument('first', type=Path)
    p.add_argument('second', type=Path)
    p.add_argument('--motion-first')
    p.add_argument('--motion-second')
    p.add_argument('--out', type=Path)
    p.set_defaults(run=join)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
