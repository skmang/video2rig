"""准备人形审阅或完整训练姿态；调用原版采样/姿态编码，不修改核心源码。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('project', type=Path)
parser.add_argument('--poses', type=int, default=24)
parser.add_argument('--seed', type=int, default=20261027)
args = parser.parse_args()
if args.poses < 24:
    parser.error('--poses must be at least 24')
ROOT = args.project.resolve()
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
raw = sample(cr, max(192, args.poses * 2), seed=args.seed, log=lambda m: print(m, flush=True))
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
    if (i + 1) % 500 == 0:
        print(f'FILTER {i + 1}/{len(raw)} accepted={len(accepted)}', flush=True)
assert len(accepted) >= args.poses - 1, f'Only {len(accepted)} usable candidates'
# 从有效姿态中按主要关节角度距离选出差异化样本，不使用图片好坏挑选。
cols = [k for j, a in rig.channels for k in [rig.index(rig.names[j], a)] if rig.names[j].startswith(('thigh_', 'calf_', 'upperarm_', 'lowerarm_', 'spine_'))]
features = (raw[accepted][:, cols] - p0[cols]) / np.maximum(hi[cols] - lo[cols], .01)
chosen = []
amplitude = np.sum(features ** 2, axis=1)
rank = np.argsort(amplitude)
groups = [rank[:int(.35 * len(rank))], rank[int(.35 * len(rank)):int(.75 * len(rank))], rank[int(.75 * len(rank)):]]
counts = (8, 8, 7) if args.poses == 24 else ((args.poses - 1) // 3, (args.poses - 1) // 3, (args.poses - 1) - 2 * ((args.poses - 1) // 3))
for group, count in zip(groups, counts):
    assert len(group) >= count, 'Not enough candidates in an amplitude group'
    local_features = features[group]
    dist = np.sum((local_features - np.median(local_features, axis=0)) ** 2, axis=1)
    for _ in range(count):
        k = int(np.argmax(dist)); chosen.append(accepted[int(group[k])])
        dist = np.minimum(dist, np.sum((local_features - local_features[k]) ** 2, axis=1))
        dist[k] = -1
params = np.vstack([p0, raw[chosen]])
# 温和档贴地，避免把腾空的平移增强误看作站姿质量；中、大幅保留原版的动态空间覆盖。
for p in params[1:1 + counts[0]]:
    mesh = rig.skin(p, cr.mesh['coords'], cr.idx, cr.wts)
    p[:3] += cr.up * (cr.floor - float((mesh @ cr.up).min()))
# 接地修正后重新核对安全边界，避免采样时通过、写出后越界。
assert not any(rejection(p) for p in params), 'A selected pose became invalid after grounding'
categories = np.array(['start_pose'] + ['mild_grounded'] * counts[0] + ['medium'] * counts[1] + ['wide'] * counts[2])
# 原版保留末尾 5% 作验证；打散幅度档，使验证集不只含最后一档。
if args.poses > 24:
    permutation = np.concatenate(([0], np.random.default_rng(args.seed + 1).permutation(np.arange(1, len(params)))))
    params, categories = params[permutation], categories[permutation]
quats, locations = pose_basis_batch(rig, params, {})
out = ROOT / 'reader'
out.mkdir(exist_ok=True)
np.savez_compressed(out / 'poses.npz', bones=np.array(rig.names), root=rig.names[rig.root], quats=quats, root_location=locations, params=params, categories=categories, seed=args.seed, prop_names=np.array([], dtype=str), prop_mats=prop_matrices(rig, params))
limits = {n: [{'axis': a, 'role': cr.role[j, a], 'lower_deg': round(float(np.degrees(lo[k])), 3), 'upper_deg': round(float(np.degrees(hi[k])), 3)} for a, k in rig.chan_of[j]] for j, n in enumerate(rig.names) if j in rig.chan_of and j != rig.root}
record = {
    'stage': 'review_poses' if args.poses == 24 else 'training_poses_ready',
    'seed': args.seed, 'candidate_count': len(raw), 'accepted_count': len(accepted),
    'rejected': rejected, 'selected_source_indices': [-1] + chosen, 'pose_count': len(params),
    'amplitude_groups': {name: (np.flatnonzero(categories == name) + 1).tolist() for name in np.unique(categories)},
    'limb_chains': [[rig.names[j] for j in ch] for ch in cr.limb_chains(True)],
    'standing_legs': [rig.names[j] for j in sorted(cr.legs)], 'limits_deg': limits,
    'coverage_note': '沿用已确认的人形输入约束；清晰渲染作为基础，训练时由原版执行颜色、噪声、分辨率和模糊增强。',
}
if args.poses > 24:
    record['selected_source_indices'] = np.array([-1] + chosen)[permutation].tolist()
(ROOT / 'Input' / 'sampling.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in record.items() if k not in ('limits_deg', 'selected_source_indices', 'amplitude_groups')}, ensure_ascii=False), flush=True)
