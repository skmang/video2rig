"""整理此次审阅的模型副本；保留网格和权重，规范骨轴并记录转换依据。"""
import json
import math
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

if '--' not in sys.argv or len(sys.argv[sys.argv.index('--') + 1:]) != 1:
    raise SystemExit('Usage: blender source.blend -b --python prepare_humanoid.py -- PROJECT')
root = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
(root / 'Input').mkdir(parents=True, exist_ok=True)
source_path = bpy.data.filepath
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
targets = [b.name for b in arm.data.bones if 'twist' in b.name.lower()]
assert len(targets) == 8
assert not any(p.constraints for p in arm.pose.bones)
assert not any(o.animation_data and o.animation_data.drivers for o in bpy.data.objects)
for name in targets:
    assert not arm.data.bones[name].children
    for obj in meshes:
        if name in obj.vertex_groups:
            group = obj.vertex_groups[name].index
            assert not any(g.group == group and g.weight > 0 for v in obj.data.vertices for g in v.groups)
rest = {b.name: [list(row) for row in b.matrix_local] for b in arm.data.bones if b.name not in targets}
deps = bpy.context.evaluated_depsgraph_get()
before = {o.name: np.array([tuple(o.matrix_world @ v.co) for v in o.evaluated_get(deps).data.vertices]) for o in meshes}
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
for name in targets:
    arm.data.edit_bones.remove(arm.data.edit_bones[name])
# FBX 的关节位置可用，但骨尾/局部轴没有沿肢体。原版依赖局部 Y 为骨长方向。
next_joint = {'pelvis': 'spine_01', 'spine_01': 'spine_02', 'spine_02': 'spine_03', 'spine_03': 'neck_01', 'neck_01': 'head'}
for side in ('l', 'r'):
    next_joint.update({f'clavicle_{side}': f'upperarm_{side}', f'upperarm_{side}': f'lowerarm_{side}', f'lowerarm_{side}': f'hand_{side}', f'hand_{side}': f'middle_01_{side}', f'thigh_{side}': f'calf_{side}', f'calf_{side}': f'foot_{side}', f'foot_{side}': f'ball_{side}'})
edit = arm.data.edit_bones
for name, child in next_joint.items():
    edit[name].tail = edit[child].head.copy()
    edit[name].align_roll(Vector((0, -1, 0)))
for side in ('l', 'r'):
    toe, foot = edit['ball_' + side], edit['foot_' + side]
    direction = toe.head - foot.head
    direction.z = 0
    toe.tail = toe.head + direction.normalized() * 7
    toe.align_roll(Vector((0, 0, 1)))
edit['head'].tail = edit['head'].head + Vector((0, 0, 16))
edit['head'].align_roll(Vector((0, -1, 0)))
bpy.ops.object.mode_set(mode='OBJECT')
for pb in arm.pose.bones:
    pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()
neutral_error = max(float(np.linalg.norm(np.array([tuple(o.matrix_world @ v.co) for v in o.evaluated_get(deps).data.vertices]) - before[o.name], axis=1).max()) for o in meshes)
assert neutral_error < 0.0001, f'Neutral mesh changed by {neutral_error} world units'
normalized = {b.name: [list(row) for row in b.matrix_local] for b in arm.data.bones}

# 仅调整起始姿势：每侧大腿外展 4 度，让双脚不再挤在一起；不应用为静止姿势。
bpy.context.view_layer.update()
centre = arm.pose.bones['pelvis'].head.copy()
pose_changes = []
for side in ('l', 'r'):
    pb = arm.pose.bones['thigh_' + side]
    foot = arm.pose.bones['foot_' + side]
    base = pb.matrix_basis.copy()
    original = foot.head.copy()
    outward = Vector((1 if original.x > centre.x else -1, 0, 0))
    best = None
    for axis in (0, 2):
        for sign in (-1, 1):
            pb.matrix_basis = base
            pb.rotation_mode = 'QUATERNION'
            v = Vector((int(axis == 0), 0, int(axis == 2)))
            pb.rotation_quaternion = base.to_quaternion() @ Quaternion(v, sign * math.radians(4))
            bpy.context.view_layer.update()
            displacement = foot.head - original
            score = displacement.dot(outward) - abs(displacement.y)
            if best is None or score > best[0]:
                best = (score, axis, sign)
    pb.matrix_basis = base
    pb.rotation_mode = 'QUATERNION'
    axis, sign = best[1:]
    pb.rotation_quaternion = base.to_quaternion() @ Quaternion(Vector((int(axis == 0), 0, int(axis == 2))), sign * math.radians(4))
    pose_changes.append({'bone': pb.name, 'local_axis': axis, 'degrees': sign * 4})
    bpy.context.view_layer.update()

scene = bpy.context.scene
scene.render.fps = 24
scene.frame_set(1)
bpy.context.preferences.filepaths.save_version = 0
output = root / 'Input' / 'MedievalGirl_Humanoid.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(output))
record = {'source': source_path, 'removed_unweighted_leaf_bones': targets, 'bone_axes_normalized': list(next_joint) + ['ball_l', 'ball_r', 'head'], 'neutral_mesh_max_world_error': neutral_error, 'pose_changes': pose_changes, 'remaining_bones': len(arm.data.bones), 'meshes': len(meshes), 'vertices': sum(len(o.data.vertices) for o in meshes), 'original_rest_matrices': rest, 'normalized_rest_matrices': normalized, 'animation_transfer': 'Bone axes changed in this processing copy. Convert local rotations using the recorded rest matrices before applying animation to the original game skeleton; do not copy raw local quaternions.'}
(root / 'Input' / 'preparation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('PREPARED', json.dumps({k: v for k, v in record.items() if not k.endswith('matrices')}), flush=True)
