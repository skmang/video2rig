"""为该人形准备少量审阅输入；调用原版采样/姿态编码，不修改第三方源码。"""
import json
from pathlib import Path
import sys
import numpy as np

if len(sys.argv) != 2:
    raise SystemExit('Usage: python ' + Path(__file__).name + ' PROJECT')
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video2rig.common import Project
from video2rig.creature import Creature
from video2rig.reader.poses import sample, pose_basis_batch, prop_matrices
from video2rig.reader import project_ortho


class HumanoidInput(Creature):
    """仅用于准备训练输入的保守活动范围；原版拟合器和训练器保持原样。"""
    def bounds(self, clip):
        lo, hi, p0 = super().bounds(clip)
        rig = self.rig
        H, _ = rig.bone_points(p0)
        for j, a in rig.channels:
            n = rig.names[j]
            if j == rig.root:
                continue
            k = rig.index(n, a)
            role = self.role[j, a]
            limits = None
            if n.startswith('calf_'):
                limits = (-130, 0) if a == self.bend_axis(j) else (-5, 5) if a == 1 else (-2, 2)
            elif n.startswith('thigh_'):
                if role == 'bend':
                    q = p0.copy(); q[k] += .05
                    h, _ = rig.bone_points(q)
                    child = rig.names.index('calf_' + n[-1])
                    positive_forward = (h[child] - H[child]) @ self.facing > 0
                    limits = (-25, 105) if positive_forward else (-105, 25)
                elif role == 'twist':
                    limits = (-15, 15)
                else:
                    q = p0.copy(); q[k] += .05
                    h, _ = rig.bone_points(q)
                    child = rig.names.index('calf_' + n[-1])
                    outward = np.sign(H[child, 0] - H[rig.root, 0])
                    positive_out = (h[child, 0] - H[child, 0]) * outward > 0
                    limits = (-8, 30) if positive_out else (-30, 8)
            elif n.startswith('lowerarm_'):
                limits = (-20, 125) if a == self.bend_axis(j) else (-60, 60) if a == 1 else (-2, 2)
            elif n.startswith('upperarm_'):
                limits = (-35, 125) if role == 'bend' else (-40, 40) if role == 'twist' else (-80, 80)
            elif n.startswith('hand_'):
                limits = (-40, 40) if role == 'bend' else (-20, 20)
            elif n.startswith(('spine_', 'clavicle_', 'neck_')):
                limits = (-15, 15)
            elif n.startswith(('hair_', 'breast_')):
                limits = (-5, 5)
            if limits is not None:
                lo[k] = max(lo[k], p0[k] + np.radians(limits[0]))
                hi[k] = min(hi[k], p0[k] + np.radians(limits[1]))
        return lo, hi, p0


pr = Project(ROOT)
cr = HumanoidInput(pr, pr.clip_cfg(''))
rig = cr.rig
assert len(cr.legs) == 8
lo, hi, p0 = cr.bounds(None)
raw = sample(cr, 192, seed=20261027)
ids = {n: rig.names.index(n) for n in rig.names}
accepted, rejected = [], {}
cameras = json.loads((ROOT / 'start' / 'cameras.json').read_text(encoding='utf-8'))


def rejection(p):
    G, H, _ = rig.fk(p)
    root_rotation = G[rig.root] @ rig.rest_rot[rig.root].T
    local = (H - H[rig.root]) @ root_rotation
    # 两条腿不互相穿过；允许前后错开，但本轮不采交叉步。
    for part in ('calf_', 'foot_'):
        if local[ids[part + 'l'], 0] - local[ids[part + 'r'], 0] < .025 * cr.height:
            return 'legs_cross_or_touch'
    # 使用实际关节中心检查上、下肢长度方向，排除反折/过度折叠。
    for side in ('l', 'r'):
        hip, knee, ankle = [local[ids[n + side]] for n in ('thigh_', 'calf_', 'foot_')]
        u, v = hip - knee, ankle - knee
        flex = 180 - np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        if flex > 140:
            return 'knee_overfold'
        # 手腕/肘部不能穿过胸腹的简化椭圆柱；这不是完整网格碰撞求解。
        for part in ('hand_', 'lowerarm_'):
            point = local[ids[part + side]]
            if .03 * cr.height < point[2] < .25 * cr.height and (point[0] / (.075 * cr.height)) ** 2 + (point[1] / (.07 * cr.height)) ** 2 < 1:
                return 'arm_inside_torso'
    mesh = rig.skin(p, cr.mesh['coords'], cr.idx, cr.wts)
    for camera in cameras['views']:
        size = (camera['width_px'], camera['height_px'])
        uv = project_ortho(camera, cr.data['armature_world'], mesh, size)
        if np.any(uv.min(0) < np.array(size) * .02) or np.any(uv.max(0) > np.array(size) * .98):
            return 'outside_camera_safety_margin'
    return None


for i, p in enumerate(raw):
    why = rejection(p)
    if why:
        rejected[why] = rejected.get(why, 0) + 1
    else:
        accepted.append(i)
assert len(accepted) >= 23, f'Only {len(accepted)} usable candidates'
# 从有效姿态中按主要关节角度距离选出差异化样本，不使用图片好坏挑选。
cols = [k for j, a in rig.channels for k in [rig.index(rig.names[j], a)] if rig.names[j].startswith(('thigh_', 'calf_', 'upperarm_', 'lowerarm_', 'spine_'))]
features = (raw[accepted][:, cols] - p0[cols]) / np.maximum(hi[cols] - lo[cols], .01)
chosen = []
amplitude = np.sum(features ** 2, axis=1)
rank = np.argsort(amplitude)
groups = [rank[:int(.35 * len(rank))], rank[int(.35 * len(rank)):int(.75 * len(rank))], rank[int(.75 * len(rank)):]]
for group, count in zip(groups, (8, 8, 7)):
    local_features = features[group]
    dist = np.sum((local_features - np.median(local_features, axis=0)) ** 2, axis=1)
    for _ in range(count):
        k = int(np.argmax(dist)); chosen.append(accepted[int(group[k])])
        dist = np.minimum(dist, np.sum((local_features - local_features[k]) ** 2, axis=1))
        dist[k] = -1
params = np.vstack([p0, raw[chosen]])
# 温和档贴地，避免把腾空的平移增强误看作站姿质量；中、大幅保留原版的动态空间覆盖。
for p in params[1:9]:
    mesh = rig.skin(p, cr.mesh['coords'], cr.idx, cr.wts)
    p[:3] += cr.up * (cr.floor - float((mesh @ cr.up).min()))
quats, locations = pose_basis_batch(rig, params, {})
out = ROOT / 'reader'
out.mkdir(exist_ok=True)
np.savez_compressed(out / 'poses.npz', bones=np.array(rig.names), root=rig.names[rig.root], quats=quats, root_location=locations, params=params, seed=20261027, prop_names=np.array([], dtype=str), prop_mats=prop_matrices(rig, params))
limits = {n: [{'axis': a, 'role': cr.role[j, a], 'lower_deg': round(float(np.degrees(lo[k])), 3), 'upper_deg': round(float(np.degrees(hi[k])), 3)} for a, k in rig.chan_of[j]] for j, n in enumerate(rig.names) if j in rig.chan_of and j != rig.root}
record = {'stage': 'awaiting_user_image_review_not_trained', 'candidate_count': len(raw), 'accepted_count': len(accepted), 'rejected': rejected, 'selected_source_indices': [-1] + chosen, 'review_poses': len(params), 'amplitude_groups': {'start_pose': [1], 'mild_grounded': list(range(2, 10)), 'medium': list(range(10, 18)), 'wide': list(range(18, 25))}, 'sharp_only_for_review': True, 'limb_chains': [[rig.names[j] for j in ch] for ch in cr.limb_chains(True)], 'standing_legs': [rig.names[j] for j in sorted(cr.legs)], 'limits_deg': limits, 'coverage_note': '24 张为输入与关节约束审阅样本，不代表完整动作类别覆盖。后续扩充须沿用此输入范围，不能直接用 prepare 的默认 poses 重新覆盖。'}
(ROOT / 'Input' / 'sampling.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in record.items() if k != 'limits_deg'}, ensure_ascii=False), flush=True)
