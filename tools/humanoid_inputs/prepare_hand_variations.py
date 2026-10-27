"""将合理手型写入原版姿态输入；用完整手指运动另算可见性，不改识别/拟合目标。"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

if len(sys.argv) != 2:
    raise SystemExit('Usage: python ' + Path(__file__).name + ' PROJECT')
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video2rig.common import Project
from video2rig.creature import Creature
from video2rig.reader import keypoints

pr = Project(ROOT)
cr = Creature(pr, pr.clip_cfg(''))
rig = cr.rig
names = rig.names
data = dict(np.load(ROOT / 'reader' / 'poses.npz'))
Q = data['quats'].copy()
presets = ['自然放松', '张掌', '半握', '抓握', '握拳']
angles = [(0, 0, 0), (-6, None, -5), (35, 40, 15), (50, 65, 25), (65, 85, 40)]
thumb_angles = [(0, 0, 0), (-5, -8, -3), (10, 12, 5), (25, 22, 10), (45, 35, 20)]
records = []
finger_ids = [j for j,n in enumerate(names) if n.startswith(('index_', 'middle_', 'ring_', 'pinky_', 'thumb_'))]
hand_quaternions = {}


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


for side in ('l', 'r'):
    curves = []
    for finger in ('index', 'middle', 'ring', 'pinky'):
        p = [rig.rest_head[names.index(f'{finger}_{i:02d}_{side}')] for i in (1,2,3)]
        a,b=unit(p[1]-p[0]),unit(p[2]-p[1]); curves.append(b-a*(a@b))
    palmward = unit(np.mean(curves, axis=0))
    table = []
    for preset in range(5):
        rotations = {}
        for finger in ('index', 'middle', 'ring', 'pinky', 'thumb'):
            joint_ids = [names.index(f'{finger}_{i:02d}_{side}') for i in (1,2,3)]
            p = rig.rest_head[joint_ids]
            initial_curve = np.degrees(np.arccos(np.clip(unit(p[1]-p[0]) @ unit(p[2]-p[1]),-1,1)))
            # 不对每根手指盲加同样弯曲量：源模型各指已经有不同的自然弯曲。
            mine = cr.mesh['coords'][np.where(cr.idx == joint_ids[-1], cr.wts, 0).sum(1) > .5]
            tip_direction = unit(mine.mean(0)-p[-1])
            projected = (mine-p[-1])@tip_direction
            tip = mine[projected >= np.percentile(projected,90)].mean(0)
            directions = [unit(p[0]-rig.rest_head[names.index('hand_'+side)]), unit(p[1]-p[0]), unit(p[2]-p[1]), unit(tip-p[2])]
            rest_angles = [np.degrees(np.arccos(np.clip(a@b,-1,1))) for a,b in zip(directions[:-1],directions[1:])]
            for index,j in enumerate(joint_ids):
                along = directions[index+1]
                toward = palmward
                if finger == 'thumb':
                    toward = unit(rig.rest_head[names.index('middle_01_'+side)]-p[index])
                axis_world = unit(np.cross(along,toward))
                axis_local = rig.rest_rot[j].T @ axis_world
                value = thumb_angles[preset][index] if finger=='thumb' else angles[preset][index]
                if value is None:
                    value = -min(initial_curve,40)
                elif finger != 'thumb' and preset >= 2:
                    value -= rest_angles[index]
                rotation = Rotation.from_rotvec(axis_local*np.radians(value))
                if preset==1 and index==0 and finger!='thumb':
                    spread = {'index':-5,'middle':0,'ring':3,'pinky':7}[finger]
                    rotation = rotation * Rotation.from_rotvec((rig.rest_rot[j].T@palmward)*np.radians(spread if side=='l' else -spread))
                xyzw=rotation.as_quat();rotations[names[j]]=xyzw[[3,0,1,2]].tolist()
        table.append(rotations)
    hand_quaternions[side]=table

for i in range(len(Q)):
    assignment = {'l':i%5,'r':i%5 if i<5 else (i+2)%5}
    for side,preset in assignment.items():
        for name,q in hand_quaternions[side][preset].items():
            Q[i,names.index(name)]=q
    records.append({'pose':i+1,'left':presets[assignment['l']],'right':presets[assignment['r']]})
data['quats']=Q
data['hand_pose_names']=np.array([[r['left'],r['right']] for r in records])
np.savez_compressed(ROOT/'reader'/'poses.npz',**data)

# 原版可见性工具以 params 驱动蒙皮。提供专用上下文，使它也看见实际弯曲的手指。
# 主项目仍用原来的拟合骨骼和关键点，未添加手指预测头或手指动作求解。
context=ROOT/'reader'/'hand_visibility'
context.mkdir(exist_ok=True)
cfg=dict(pr.clip_cfg(''))
cfg.update(name='HandVisibilityOnly',rig=str(ROOT/'rig'),start=str(ROOT/'start'))
fitted=set(j for j,_ in rig.channels)|set(finger_ids)
cfg['fit_bones']=[n for j,n in enumerate(names) if j in fitted]
(context/'creature.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
full=Creature(Project(context),cfg)
assert keypoints(full)==keypoints(cr)
parameters=[]
for q,loc in zip(Q,data['root_location']):
    basis=np.repeat(np.eye(4)[None],rig.nb,axis=0)
    basis[:,:3,:3]=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    basis[rig.root,:3,3]=loc
    parameters.append(full.rig.params_from_basis(basis,names))
full_data=dict(data);full_data['params']=np.array(parameters)
np.savez_compressed(context/'poses.npz',**full_data)
record={'presets':presets,'poses':records,'quaternion_presets':hand_quaternions,'purpose':'手型外观增强及审阅；原版仍不训练逐指识别或逐指动作拟合。','visibility_context':str(context),'fitted_body_keypoints_unchanged':True}
(ROOT/'Input'/'hand-poses.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('HAND_VARIATIONS',len(Q),'poses, 5 presets; original body keypoints unchanged',flush=True)
