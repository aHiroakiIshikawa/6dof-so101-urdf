#!/usr/bin/env python3
"""Offline FK->IK round-trip benchmark. Does not connect to robot hardware.
Run from the LeRobot uv environment. Joint API units: degrees; URDF units: rad/m.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from lerobot.model.kinematics import RobotKinematics

JOINTS = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_yaw','wrist_roll']

def pose_error(actual, target):
    position_mm = float(np.linalg.norm(actual[:3,3]-target[:3,3])*1000)
    relative = target[:3,:3].T @ actual[:3,:3]
    angle_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
    return position_mm, angle_deg

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--urdf',type=Path,default=Path(__file__).with_name('so101_6dof_new_calib.urdf'))
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--samples',type=int,default=30,help='Random targets; each tested from near and zero seeds.')
    ap.add_argument('--seed',type=int,default=20260928)
    ap.add_argument('--max-iters',type=int,default=100)
    ap.add_argument('--restarts',type=int,default=5,help='Extra random seeds after first attempt fails.')
    ap.add_argument('--visualize',action='store_true',help='Step through the solved five reference poses in MeshCat.')
    args=ap.parse_args()
    if args.samples<1 or args.max_iters<1 or args.restarts<0:ap.error('Invalid count')
    urdf=args.urdf.expanduser().resolve();out=(args.output_dir or urdf.parent/'ik_test_results').resolve();out.mkdir(parents=True,exist_ok=True)
    xml=ET.parse(urdf).getroot()
    limits=[xml.find(f"joint[@name='{name}']/limit") for name in JOINTS]
    lower=np.rad2deg([float(l.get('lower')) for l in limits]);upper=np.rad2deg([float(l.get('upper')) for l in limits]);span=upper-lower
    rng=np.random.default_rng(args.seed)
    kin=RobotKinematics(str(urdf),target_frame_name='gripper_frame_link',joint_names=JOINTS)
    # Explicitly enforce position limits and freeze gripper; these are benchmark settings.
    kin.solver.enable_joint_limits(True)
    kin.solver.mask_dof('gripper')
    kin.robot.set_joint('gripper',0.0)
    cases=[]
    for name,q in [('zero',[0]*6),('yaw_plus_30',[0,0,0,0,30,0]),('yaw_minus_30',[0,0,0,0,-30,0]),('roll_plus_30',[0,0,0,0,0,30]),('roll_minus_30',[0,0,0,0,0,-30])]:
        cases.append((name,'reference',np.array(q,dtype=float),np.array([15,-20,25,-15,10,20],dtype=float)))
    for i in range(args.samples):
        q=rng.uniform(lower+.1*span,upper-.1*span)
        near=np.clip(q+rng.uniform(-15,15,6),lower+.01*span,upper-.01*span)
        cases.extend([(f'random_{i:03d}_near','near_seed',q,near),(f'random_{i:03d}_zero','zero_seed',q,np.zeros(6))])
    def attempt(target,seed,iters):
        started=time.perf_counter()
        try:
            # LeRobot sets joints in inverse_kinematics; explicitly refresh cached FK first.
            kin.forward_kinematics(seed)
            q=kin.inverse_kinematics(seed.copy(),target.copy(),position_weight=1.0,orientation_weight=.01,max_iters=iters)
            if not np.isfinite(q).all():raise ValueError('Non-finite IK result')
            actual=kin.forward_kinematics(q).copy();pe,ae=pose_error(actual,target)
            within=bool(np.all(q>=lower-1e-5) and np.all(q<=upper+1e-5))
            return {'passed':pe<=1.0 and ae<=1.0 and within,'q_deg':q.tolist(),'position_error_mm':pe,'orientation_error_deg':ae,'within_limits':within,'time_ms':(time.perf_counter()-started)*1000}
        except Exception as exc:
            return {'passed':False,'error':str(exc),'time_ms':(time.perf_counter()-started)*1000}
    rows=[]
    for index,(name,group,q,seed) in enumerate(cases):
        target=kin.forward_kinematics(q).copy()
        default=attempt(target,seed,8);long=attempt(target,seed,args.max_iters)
        attempts=[long]
        for _ in range(args.restarts):
            if attempts[-1]['passed']:break
            restart_seed=rng.uniform(lower+.1*span,upper-.1*span)
            r=attempt(target,restart_seed,args.max_iters);r['seed_deg']=restart_seed.tolist();attempts.append(r)
        robust=next((x for x in attempts if x['passed']),min(attempts,key=lambda x:x.get('position_error_mm',1e10)+x.get('orientation_error_deg',1e10)))
        rows.append({'name':name,'group':group,'target_q_deg':q.tolist(),'initial_seed_deg':seed.tolist(),'target_T':target.tolist(),'iters_8':default,'iters_long':long,'robust':robust,'robust_attempts':attempts})
        if index%10==0:print(f'Progress {index+1}/{len(cases)}',flush=True)
    unreachable=np.eye(4);unreachable[:3,3]=[2,0,1]
    rejected=attempt(unreachable,np.zeros(6),args.max_iters)
    summary={}
    for group in ['reference','near_seed','zero_seed','all']:
        selected=[r for r in rows if group=='all' or r['group']==group]
        summary[group]={'total':len(selected),**{mode:sum(r[mode]['passed'] for r in selected) for mode in ['iters_8','iters_long','robust']}}
    passed=[r['robust'] for r in rows if r['robust']['passed']]
    report={'urdf':str(urdf),'urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest(),'joint_order':JOINTS,'joint_lower_deg':lower.tolist(),'joint_upper_deg':upper.tolist(),'settings':vars(args)|{'urdf':str(urdf),'output_dir':str(out)},'thresholds':{'position_mm':1,'orientation_deg':1,'joint_limits':True},'versions':{n:importlib.metadata.version(n) for n in ['numpy','placo','lerobot']},'summary':summary,'unreachable':rejected,'max_passed_position_error_mm':max((r['position_error_mm'] for r in passed),default=None),'max_passed_orientation_error_deg':max((r['orientation_error_deg'] for r in passed),default=None),'cases':rows,'scope':'Offline pose IK only; no collision avoidance, trajectory validation, singularity guarantee, or hardware test. Targets are FK-generated within the inner 80 percent of each joint range.'}
    (out/'ik_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# SO-101 6DoF IKテスト結果','',f'URDF SHA256: `{report["urdf_sha256"]}`','',f'合格基準: 位置誤差≤1 mm、姿勢誤差≤1°、関節制限内。乱数seed={args.seed}。','',f'LeRobot inverse_kinematicsを使用。関節制限を明示的に有効化し、gripperを0に固定。位置/姿勢weight=1.0/0.01。','',f'| テスト | 件数 | 8反復 | {args.max_iters}反復 | 再初期化込み（追加最大{args.restarts}回） |','|---|---:|---:|---:|---:|']
    for group,s in summary.items():lines.append(f'| {group} | {s["total"]} | {s["iters_8"]} | {s["iters_long"]} | {s["robust"]} |')
    lines+=['',f'合格解の最大位置誤差: {report["max_passed_position_error_mm"]} mm',f'合格解の最大姿勢誤差: {report["max_passed_orientation_error_deg"]}°',f'到達不能目標(2,0,1)mの不合格判定: {not rejected["passed"]}','', 'referenceはzero/yaw±30°/roll±30°を別の初期姿勢から解く。near_seedは既知解から±15°の初期姿勢、zero_seedは全関節0から解く。目標生成に使った関節角はIKへ渡さず、解のFKと目標姿勢を比較する。','', 'この結果は無作為に抽出した到達可能姿勢の運動学テスト。衝突回避、関節速度、経路の連続性、全作業範囲、実機精度は保証しない。FK/IKとも同じURDFを使うため、CADや実機との絶対精度の独立検証でもない。','', '## 参照5姿勢の解','', '| 目標 | 合否 | 位置誤差mm | 姿勢誤差deg | 解の関節角deg（ID1〜ID6） |','|---|---|---:|---:|---|']
    for row in rows[:5]:
        z=row['robust'];lines.append(f'| {row["name"]} | {z["passed"]} | {z.get("position_error_mm",float("nan")):.6g} | {z.get("orientation_error_deg",float("nan")):.6g} | {np.round(z.get("q_deg",[]),3).tolist()} |')
    (out/'IK_TEST_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'summary':summary,'unreachable_rejected':not rejected['passed'],'max_position_mm':report['max_passed_position_error_mm'],'max_orientation_deg':report['max_passed_orientation_error_deg']},indent=2),flush=True)
    if args.visualize:
        from placo_utils.visualization import robot_viz,robot_frame_viz
        viz=robot_viz(kin.robot)
        for row in rows[:5]:
            result=row['robust']
            if not result['passed']:continue
            kin.forward_kinematics(np.array(result['q_deg']));viz.display(kin.robot.state.q);robot_frame_viz(kin.robot,'gripper_frame_link',scale=.06)
            print(row['name'],result['q_deg']);input('Enterで次のIK解へ（Ctrl-Cで終了）: ')
    return 0 if summary['all']['robust']==len(rows) and not rejected['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
