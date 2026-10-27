"""Blender 后台入口：角色基准、姿势捕获、固定相机渲染、原骨架回传。"""
import argparse
import math
import os
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import digest, file_hash, identifier, library, pose, processing_basis, read, write


def armature(name=''):
    arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if name:
        return next(o for o in arms if o.name == name)
    if len(arms) != 1:
        raise ValueError('场景不是单骨架，请使用 --armature')
    return arms[0]


def matrix(m):
    return [list(row) for row in m]


def source_space(arm, c):
    for label, rest in (('processing', {b['name']: b['rest'] for b in c['rig']['bones']}),
                        ('original', c['original_rest'])):
        if all(n in arm.data.bones and np.allclose(arm.data.bones[n].matrix_local, r, atol=1e-4)
               for n, r in rest.items()):
            extra = set(arm.data.bones.keys()) - set(rest)
            expected = set(c['removed_unweighted_leaf_bones']) if label == 'original' else set()
            if extra != expected:
                raise ValueError('骨架包含未记录的新增或缺失骨骼')
            if any(arm.data.bones[n].children for n in extra):
                raise ValueError('被移除辅助骨不再是叶骨')
            for obj in mesh_objects(arm):
                groups = {g.index for g in obj.vertex_groups if g.name in extra}
                if any(g.group in groups and g.weight > 0 for v in obj.data.vertices for g in v.groups):
                    raise ValueError('原骨架辅助骨已有蒙皮权重，不能继续按旧映射转换')
            for b in c['rig']['bones']:
                bone = arm.data.bones[b['name']]
                parent = bone.parent.name if bone.parent else None
                if parent != b['parent'] or bone.inherit_scale != 'FULL' or not bone.use_inherit_rotation:
                    raise ValueError('骨架层级或继承规则不配套')
            return label
    raise ValueError('当前文件的骨架静止矩阵不属于该角色基准')


def mesh_objects(arm):
    return [o for o in bpy.context.scene.objects if o.type == 'MESH' and any(
        m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]


def vertices(objects):
    deps = bpy.context.evaluated_depsgraph_get()
    arrays = []
    for obj in objects:
        ev = obj.evaluated_get(deps)
        me = ev.to_mesh()
        try:
            coords = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get('co', coords)
            m = np.array(obj.matrix_world)
            arrays.append(coords.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
        finally:
            ev.to_mesh_clear()
    if not arrays:
        raise ValueError('没有角色网格')
    return np.concatenate(arrays)


def init(a):
    root = a.library.resolve()
    if root.exists():
        raise ValueError('基准目录已存在，请使用新目录')
    project = a.project.resolve()
    rig = read(project / 'rig' / 'rig.json')
    preparation = read(project / 'Input' / 'preparation.json')
    arm = armature(a.armature)
    c = {'version': 1, 'id': identifier(a.id), 'rig': rig,
         'original_rest': preparation['original_rest_matrices'],
         'cameras': read(project / 'start' / 'cameras.json'),
         'video_size': [1024, 576], 'project': os.path.relpath(project, root).replace('\\', '/'),
         'source_blend': os.path.relpath(bpy.data.filepath, root).replace('\\', '/'),
         'source_sha256': file_hash(bpy.data.filepath),
         'source_rig_sha256': file_hash(project / 'rig' / 'rig.json'),
         'source_camera_sha256': file_hash(project / 'start' / 'cameras.json'),
         'rig_assets': {p.name: file_hash(p) for p in (project / 'rig').iterdir() if p.is_file()},
         'removed_unweighted_leaf_bones': preparation['removed_unweighted_leaf_bones']}
    cameras = c['cameras']
    views = cameras['views']
    if (len(views) != 2 or [v['name'] for v in views] != ['v2r_front', 'v2r_side']
            or any(v['type'] != 'ortho' for v in views)
            or not np.allclose(views[0]['up'], views[1]['up'], atol=1e-5)
            or not np.isclose(np.dot(views[0]['forward'], views[1]['forward']), 0, atol=1e-5)
            or not np.isclose(views[0]['ortho_height'], views[1]['ortho_height'])):
        raise ValueError('当前工作流需要等高等尺度、相差九十度的正面与侧面正交相机')
    if (sum(v['width_px'] for v in views) != cameras['frame_px'][0]
            or any(v['height_px'] != cameras['frame_px'][1] for v in views)
            or views[0]['x0_px'] != 0 or views[1]['x0_px'] != views[0]['width_px']):
        raise ValueError('相机分栏与画幅元数据不一致')
    if source_space(arm, c) != 'processing':
        raise ValueError('init 请打开经过处理的角色模型')
    if not np.allclose(arm.matrix_world, rig['armature_world'], atol=1e-5):
        raise ValueError('处理模型的物体变换与导出骨架不一致')
    for n, basis in rig['start_pose'].items():
        if not np.allclose(arm.pose.bones[n].matrix_basis, basis, atol=1e-4):
            raise ValueError('init 须使用标定时的起始姿势')
    pts = vertices(mesh_objects(arm))
    c['floor_z'] = float(pts[:, 2].min())
    c['studio_extent'] = float(np.linalg.norm(pts.max(0) - pts.min(0))) * 10
    c['fingerprint'] = digest(c)
    root.mkdir(parents=True)
    write(root / 'character.json', c)
    print('KEYFRAMES_INIT', root, c['fingerprint'], flush=True)


def snapshot(arm, c):
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    deform = {}
    for b in c['rig']['bones']:
        n = b['name']
        deform[n] = matrix(ev.pose.bones[n].matrix @ arm.data.bones[n].matrix_local.inverted())
    props = {p['name']: matrix(bpy.context.scene.objects[p['name']].matrix_world) for p in c['rig']['props']}
    return {'armature_world': matrix(arm.matrix_world), 'deform': deform, 'props': props}


def capture(a):
    root, c = library(a.library)
    name = identifier(a.id)
    out = root / 'poses' / name
    if out.exists():
        raise ValueError('姿势 ID 已存在，不原位覆盖被引用的姿势')
    arm = armature(a.armature)
    space = source_space(arm, c)
    if not set(a.contacts) <= {b['name'] for b in c['rig']['bones']}:
        raise ValueError('接触骨骼名称不存在')
    scene = bpy.context.scene
    scene.frame_set(a.frame)
    bpy.context.view_layer.update()
    centre = snapshot(arm, c)
    window = []
    fps = scene.render.fps / scene.render.fps_base
    for f in range(a.frame - a.window, a.frame + a.window + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        window.append({'offset_seconds': (f - a.frame) / fps, **snapshot(arm, c)})
    scene.frame_set(a.frame)
    p = {'version': 1, 'id': name, 'character': c['fingerprint'], 'purpose': a.purpose,
         'source': os.path.relpath(bpy.data.filepath, root).replace('\\', '/'),
         'source_sha256': file_hash(bpy.data.filepath), 'source_space': space,
         'source_frame': a.frame, 'source_fps': fps, 'note': a.note,
         'contacts': a.contacts, 'sample': centre, 'boundary_window': window,
         'processing_basis': processing_basis(c, centre)}
    p['fingerprint'] = digest(p)
    out.mkdir(parents=True)
    write(out / 'pose.json', p)
    print('KEYFRAMES_CAPTURE', out, space, flush=True)


def apply_sample(arm, c, sample):
    if any(p.constraints for p in arm.pose.bones):
        raise ValueError('写入目标骨架存在约束；请使用可烘焙的变形骨架副本')
    if arm.animation_data:
        arm.animation_data.action = None
        for track in arm.animation_data.nla_tracks:
            track.mute = True
        if arm.animation_data.drivers:
            raise ValueError('写入目标骨架存在驱动，不直接覆盖')
    arm.matrix_world = Matrix(sample['armature_world'])
    desired = {n: Matrix(d) @ arm.data.bones[n].matrix_local for n, d in sample['deform'].items()}
    for b in c['rig']['bones']:
        n = b['name']
        bone = arm.data.bones[n]
        kwargs = {}
        if bone.parent:
            kwargs = {'parent_matrix': desired[bone.parent.name], 'parent_matrix_local': bone.parent.matrix_local}
        arm.pose.bones[n].matrix_basis = bone.convert_local_to_pose(
            desired[n], bone.matrix_local, invert=True, **kwargs)
    for name, world in sample.get('props', {}).items():
        bpy.context.scene.objects[name].matrix_world = Matrix(world)
    bpy.context.view_layer.update()
    error = max(float(np.max(np.abs(np.array(arm.pose.bones[n].matrix) - np.array(m)))) for n, m in desired.items())
    if error > 1e-3:
        raise ValueError(f'姿势写回误差超出容差：{error}')
    return error


def render(a):
    root, c = library(a.library)
    p = pose(root, c, a.id)
    directory = root / 'poses' / p['id']
    if (directory / 'frame.png').exists() or (directory / 'render.json').exists():
        raise ValueError('该姿势已经渲染，使用新姿势 ID，不覆盖现有引用')
    if file_hash(bpy.data.filepath) != c['source_sha256']:
        raise ValueError('渲染必须打开基准绑定的处理模型文件')
    arm = armature(a.armature)
    if source_space(arm, c) != 'processing':
        raise ValueError('渲染须使用处理骨架')
    apply_sample(arm, c, p['sample'])
    objects = mesh_objects(arm) + [bpy.context.scene.objects[x['name']] for x in c['rig']['props']]
    points = vertices(objects)
    coverage = []
    for cam in c['cameras']['views']:
        if cam['type'] != 'ortho':
            raise ValueError('当前工具只处理固定正交双视角')
        delta = points - np.array(cam['location'])
        u = delta @ np.array(cam['right']) / cam['ortho_width'] + .5
        v = .5 - delta @ np.array(cam['up']) / cam['ortho_height']
        margin = float(min(u.min(), v.min(), 1 - u.max(), 1 - v.max()))
        coverage.append({'view': cam['name'], 'minimum_margin_fraction': margin})
    if min(x['minimum_margin_fraction'] for x in coverage) < 0:
        raise ValueError('姿势网格超出固定相机画幅；不要单独自动缩放')
    # 新渲染场景只链接当前角色，不改源文件，不删除源场景对象。
    scene = bpy.data.scenes.new('KeyframeMotionRender')
    for obj in [arm, *objects]:
        scene.collection.objects.link(obj)
    bpy.context.window.scene = scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = a.samples
    scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 8
    scene.render.use_motion_blur = False
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.view_transform = 'AgX'
    world = bpy.data.worlds.new('KeyframeMotionWorld')
    world.use_nodes = True
    grey = (.42, .41, .40, 1)
    world.node_tree.nodes['Background'].inputs['Color'].default_value = grey
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .9
    scene.world = world
    for name, z, energy in (('Key', -35, 3.2), ('Fill', 140, 1.2)):
        data = bpy.data.lights.new(name, 'SUN')
        data.energy, data.angle = energy, math.radians(12)
        obj = bpy.data.objects.new(name, data)
        obj.rotation_euler = (math.radians(50 if name == 'Key' else 60), 0, math.radians(z))
        scene.collection.objects.link(obj)
    extent, floor = c['studio_extent'], c['floor_z']
    me = bpy.data.meshes.new('KeyframeFloor')
    me.from_pydata([(-extent, -extent, floor), (extent, -extent, floor),
                   (extent, extent, floor), (-extent, extent, floor)], [], [(0, 1, 2, 3)])
    mat = bpy.data.materials.new('KeyframeGrey')
    mat.use_nodes = True
    shader = mat.node_tree.nodes['Principled BSDF']
    shader.inputs['Base Color'].default_value = grey
    shader.inputs['Roughness'].default_value = .9
    me.materials.append(mat)
    scene.collection.objects.link(bpy.data.objects.new('KeyframeFloor', me))
    width, height = c['cameras']['frame_px']
    pixels = np.zeros((height, width, 4), np.float32)
    for rec in c['cameras']['views']:
        data = bpy.data.cameras.new(rec['name'])
        data.type, data.sensor_fit, data.ortho_scale = 'ORTHO', 'VERTICAL', rec['ortho_height']
        data.clip_end = 1000
        obj = bpy.data.objects.new(rec['name'], data)
        scene.collection.objects.link(obj)
        rotation = Matrix((rec['right'], rec['up'], [-x for x in rec['forward']])).transposed()
        obj.matrix_world = Matrix.Translation(rec['location']) @ rotation.to_4x4()
        scene.camera = obj
        scene.render.resolution_x, scene.render.resolution_y = rec['width_px'], rec['height_px']
        path = directory / (rec['name'] + '.png')
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(path), check_existing=False)
        part = np.array(img.pixels[:], np.float32).reshape(height, rec['width_px'], 4)
        x = rec['x0_px']
        pixels[:, x:x + rec['width_px']] = part
        bpy.data.images.remove(img)
    for rec in c['cameras']['views'][1:]:
        x = rec['x0_px']
        pixels[:, x - 2:x + 2, :3] = .03
    img = bpy.data.images.new('KeyframePair', width, height, alpha=True)
    img.pixels.foreach_set(pixels.ravel())
    img.filepath_raw = str(directory / 'frame.png')
    img.file_format = 'PNG'
    img.save()
    write(directory / 'render.json', {'pose': p['fingerprint'], 'camera': digest(c['cameras']),
          'image_sha256': file_hash(directory / 'frame.png'), 'inside_frame': True,
          'coverage': coverage, 'samples': a.samples, 'engine': 'CYCLES',
          'minimum_mesh_height_above_floor': float(points[:, 2].min() - floor),
          'visual_review': 'not_assessed_by_renderer'})
    print('KEYFRAMES_RENDER', directory / 'frame.png', flush=True)


def materialize(a):
    root, c = library(a.library)
    if file_hash(bpy.data.filepath) != c['source_sha256']:
        raise ValueError('请打开基准绑定的处理模型文件')
    if a.out.exists():
        raise ValueError('输出文件已经存在，不覆盖')
    p = pose(root, c, a.id)
    arm = armature(a.armature)
    apply_sample(arm, c, p['sample'])
    bpy.context.scene.frame_set(1)
    bpy.context.scene.render.fps = 24
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out.resolve()))
    print('KEYFRAMES_MATERIALIZE', a.out, flush=True)


def transfer(a):
    root, c = library(a.library)
    arm = armature(a.armature)
    if source_space(arm, c) != 'original':
        raise ValueError('回传必须打开原骨架模型')
    if a.out.exists():
        raise ValueError('输出文件已经存在，不覆盖')
    if any(pb.constraints for pb in arm.pose.bones):
        raise ValueError('原骨架有约束，须先准备专用烘焙副本')
    if a.out.with_suffix('.transfer.json').exists():
        raise ValueError('输出报告已经存在，不覆盖')
    if a.pose:
        p = pose(root, c, a.pose)
        samples = [p['sample']]
        times = [0.0]
        action_name = p['id']
    else:
        # Blender 自带 numpy，不依赖额外 scipy；复原 quaternion FK。
        from mathutils import Quaternion
        job = a.job.resolve()
        metadata = read(job / 'prepared.json')
        if metadata['character'] != c['fingerprint']:
            raise ValueError('动作包不属于该角色')
        rig = read(job / 'rig' / 'rig.json')
        with np.load(a.motion, allow_pickle=False) as archive:
            d = dict(archive)
        if len(d['props']):
            raise ValueError('当前回传仅覆盖角色骨架；带道具轨道须单独接入，不能丢弃')
        names = [str(n) for n in d['bones']]
        times = d['times']
        if not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError('动画时间为空或没有递增')
        if not set(names) <= {b['name'] for b in rig['bones']}:
            raise ValueError('动画骨骼与动作包不配套')
        samples = []
        for i in range(len(times)):
            basis = {n: Matrix(m) for n, m in rig['start_pose'].items()}
            for j, n in enumerate(names):
                basis[n] = Quaternion(d['quaternions'][i, j]).to_matrix().to_4x4()
            basis[str(d['root'])].translation = d['root_location'][i]
            posed, deform = {}, {}
            for b in rig['bones']:
                n, parent = b['name'], b['parent']
                rest = Matrix(b['rest'])
                if parent:
                    parent_rest = Matrix(next(x['rest'] for x in rig['bones'] if x['name'] == parent))
                    posed[n] = posed[parent] @ parent_rest.inverted() @ rest @ basis[n]
                else:
                    posed[n] = rest @ basis[n]
                deform[n] = matrix(posed[n] @ rest.inverted())
            samples.append({'deform': deform, 'armature_world': d['armature_world'].tolist(), 'props': {}})
        action_name = read(job / 'action.json')['id']
    apply_sample(arm, c, samples[0])
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = 24, 1
    action = bpy.data.actions.new(action_name)
    action.use_fake_user = True
    arm.animation_data_create()
    last = {}
    errors = []
    for t, sample in zip(times, samples):
        # apply_sample 在写入前解除旧 Action，随后将烘焙键写入同一新 Action。
        errors.append(apply_sample(arm, c, sample))
        arm.animation_data.action = action
        frame = 1 + float(t) * 24
        for b in c['rig']['bones']:
            pb = arm.pose.bones[b['name']]
            pb.rotation_mode = 'QUATERNION'
            if pb.name in last and pb.rotation_quaternion.dot(last[pb.name]) < 0:
                pb.rotation_quaternion.negate()
            last[pb.name] = pb.rotation_quaternion.copy()
            for channel in ('location', 'rotation_quaternion', 'scale'):
                pb.keyframe_insert(channel, frame=frame)
    # 处理前保留静态物体变换；拟合轨道的根运动在骨骼内。
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'
    scene.frame_start = 1
    scene.frame_end = max(1, int(math.ceil(1 + float(times[-1]) * 24)))
    scene.frame_set(1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out.resolve()))
    write(a.out.with_suffix('.transfer.json'), {'character': c['fingerprint'], 'action': action_name,
          'frames': len(samples), 'max_pose_matrix_error': max(errors),
          'method': 'P_target = (P_processing @ inverse(R_processing)) @ R_original',
          'scope': 'recorded bones, all supplied frames; no Unity verification',
          'output_sha256': file_hash(a.out)})
    print('KEYFRAMES_TRANSFER', a.out, 'max_error', max(errors), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('init')
    p.add_argument('library', type=Path)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.set_defaults(run=init)
    p = sub.add_parser('capture')
    p.add_argument('library', type=Path)
    p.add_argument('--id', required=True)
    p.add_argument('--frame', type=int, default=1)
    p.add_argument('--window', type=int, choices=range(0, 9), default=0)
    p.add_argument('--purpose', choices=['action', 'calibration'], default='action')
    p.add_argument('--contacts', nargs='*', default=[])
    p.add_argument('--note', default='')
    p.set_defaults(run=capture)
    p = sub.add_parser('render')
    p.add_argument('library', type=Path)
    p.add_argument('--id', required=True)
    p.add_argument('--samples', type=int, default=16)
    p.set_defaults(run=render)
    p = sub.add_parser('materialize')
    p.add_argument('library', type=Path)
    p.add_argument('--id', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(run=materialize)
    p = sub.add_parser('transfer')
    p.add_argument('library', type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--pose')
    group.add_argument('--motion', type=Path)
    p.add_argument('--job', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(run=transfer)
    for p in sub.choices.values():
        p.add_argument('--armature', default='')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.command == 'transfer' and args.motion and not args.job:
        parser.error('--motion 需要配套 --job')
    args.run(args)


if __name__ == '__main__':
    main()
