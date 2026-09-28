#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=3.2", "numpy", "opencv-python"]
# ///
"""SO-101 6DoF wrist-camera image-based visual servoing (IBVS) in MuJoCo.

The URDF is loaded into MuJoCo, a pinhole camera is attached to the gripper,
and a 4-marker target on the table is detected by colour in the rendered wrist
image. Joint velocities from the IBVS law are integrated kinematically (no
dynamics), because the modified wrist links have no measured inertia.

    uv run visual_servo_sim.py            # results in ./visual_servo_results/
    uv run visual_servo_sim.py --show     # also show a live OpenCV window
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import math
import re
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])  # OpenCV optical frame (z forward, y down) <-> MuJoCo camera frame
PLATE_HALF, PLATE_THICK, MARKER_R = 0.035, 0.002, 0.006
# name: (rgba, corner sign in target frame, OpenCV HSV ranges). Hues avoid the yellow robot plastic.
MARKERS = {
    "red": ((0.90, 0.05, 0.05, 1), (1, 1), [((0, 120, 60), (8, 255, 255)), ((172, 120, 60), (180, 255, 255))]),
    "green": ((0.05, 0.80, 0.10, 1), (-1, 1), [((45, 120, 60), (75, 255, 255))]),
    "blue": ((0.05, 0.20, 0.95, 1), (-1, -1), [((105, 120, 60), (130, 255, 255))]),
    "magenta": ((0.90, 0.05, 0.90, 1), (1, -1), [((140, 120, 60), (165, 255, 255))]),
}
MARKER_SPACING = 0.02  # marker centres at (+-s, +-s) in the target frame


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).ravel())
    return q


def rotvec(R):
    c = float(np.clip((np.trace(R) - 1) / 2, -1, 1))
    a = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if math.sin(a) > 1e-6:
        return a * w / (2 * math.sin(a))
    if c > 0:
        return 0.5 * w
    B = (R + np.eye(3)) / 2  # angle ~ pi: R + I = 2 n n^T
    col = B[:, int(np.argmax(np.linalg.norm(B, axis=0)))]
    return a * col / np.linalg.norm(col)


def dls(J, e, damping):
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)


def target_points_world(target_xy, target_yaw):
    R = rot_z(target_yaw)
    top = np.array([target_xy[0], target_xy[1], PLATE_THICK])
    return {n: top + R @ np.array([sx * MARKER_SPACING, sy * MARKER_SPACING, 0]) for n, (_, (sx, sy), _) in MARKERS.items()}


def goal_camera_pose(target_xy, distance, pitch, goal_yaw):
    """Camera on the robot side of the target, looking at it with `pitch` below horizontal (90 = straight down).

    Image 'down' points toward the robot base.
    """
    radial = np.array([target_xy[0], target_xy[1], 0.0])
    radial /= np.linalg.norm(radial)
    z = math.cos(pitch) * radial + np.array([0.0, 0.0, -math.sin(pitch)])
    x = np.cross(radial, [0, 0, 1])
    R = np.column_stack([x, np.cross(z, x), z]) @ rot_z(goal_yaw)
    return np.array([target_xy[0], target_xy[1], PLATE_THICK]) - distance * z, R


def build_model(args, goal_p, goal_R):
    text = args.urdf.read_text()
    compiler = (
        f'<mujoco><compiler meshdir="{args.urdf.parent / "assets"}" strippath="true" fusestatic="false" '
        'discardvisual="false" boundmass="1e-4" boundinertia="1e-8"/></mujoco>'
    )
    text = re.sub(r"(<robot\b[^>]*>)", lambda m: m.group(1) + compiler, text, count=1)
    spec = mujoco.MjSpec.from_string(text)
    spec.visual.global_.offwidth = max(args.width, 640) * 2
    spec.visual.global_.offheight = max(args.height, 480) * 2

    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.55, 0.55, 0.55], rgb2=[0.42, 0.42, 0.42], width=512, height=512)
    textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"
    spec.add_material(name="grid", textures=textures, texrepeat=[10, 10], specular=0.1)
    world = spec.worldbody
    world.add_light(pos=[0.9, 0.4, 1.2], dir=[-0.6, -0.3, -1], diffuse=[0.6] * 3, specular=[0.1] * 3, castshadow=True)
    world.add_light(pos=[-0.5, -0.5, 1.0], dir=[0.5, 0.5, -1], diffuse=[0.35] * 3, specular=[0] * 3, castshadow=False)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], material="grid")

    target = world.add_body(name="target", pos=[*args.target_xy, 0], quat=mat2quat(rot_z(math.radians(args.target_yaw_deg))),
                            mocap=True)
    target.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[PLATE_HALF, PLATE_HALF, PLATE_THICK / 2],
                    pos=[0, 0, PLATE_THICK / 2], rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)
    for name, (rgba, (sx, sy), _) in MARKERS.items():
        spec.add_material(name=f"mk_{name}", rgba=list(rgba), specular=0.0, shininess=0.0)
        target.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[MARKER_R, 0.0002, 0], material=f"mk_{name}",
                        pos=[sx * MARKER_SPACING, sy * MARKER_SPACING, PLATE_THICK + 0.0002], contype=0, conaffinity=0)

    # Wrist camera in gripper_frame_link (z = approach direction). Tilted by cam_tilt toward the tool axis.
    R_gc = rot_x(math.copysign(math.radians(args.cam_tilt_deg), args.cam_offset[0]))
    p_gc = np.array([0.0, args.cam_offset[0], -args.cam_offset[1]])
    grip = spec.body("gripper_frame_link")
    grip.add_camera(name="wrist_cam", pos=p_gc, quat=mat2quat(R_gc @ CV_TO_MJ), fovy=args.fovy)
    # Group-3 geoms are drawn only in the external view.
    grip.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.016, 0.016, 0.006], pos=p_gc - R_gc[:, 2] * 0.007,
                  quat=mat2quat(R_gc), rgba=[0.2, 0.2, 0.25, 1], group=3, contype=0, conaffinity=0)
    ghost = world.add_body(name="goal_ghost", pos=goal_p, quat=mat2quat(goal_R), mocap=True)
    ghost.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.016, 0.016, 0.006], pos=[0, 0, -0.007],
                   rgba=[0.1, 0.9, 0.2, 0.35], group=3, contype=0, conaffinity=0)
    return spec.compile()


class Sim:
    def __init__(self, model, width, height):
        self.m, self.d = model, mujoco.MjData(model)
        jids = [model.joint(n).id for n in ARM_JOINTS]
        self.qadr, self.dadr = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
        self.lo, self.hi = model.jnt_range[jids].T.copy()
        self.cam = model.camera("wrist_cam").id
        self.cam_body = model.cam_bodyid[self.cam]
        self.floor = model.geom("floor").id
        self.f = 0.5 * height / math.tan(math.radians(model.cam_fovy[self.cam]) / 2)
        self.c = np.array([(width - 1) / 2, (height - 1) / 2])
        self.wrist = mujoco.Renderer(model, height, width)
        self.ext = mujoco.Renderer(model, 480, 640)
        self.opt_ext = mujoco.MjvOption()
        self.opt_ext.geomgroup[3] = 1
        self.opt_ext.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1
        self.ext_cam = mujoco.MjvCamera()
        self.ext_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def set_q(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.d)

    def q(self):
        return self.d.qpos[self.qadr].copy()

    def cam_pose(self):
        return self.d.cam_xpos[self.cam].copy(), self.d.cam_xmat[self.cam].reshape(3, 3) @ CV_TO_MJ

    def jacobian(self):
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jp, jr, self.d.cam_xpos[self.cam], self.cam_body)
        return np.vstack([jp[:, self.dadr], jr[:, self.dadr]])

    def floor_contacts(self):
        bodies = set()
        for c in self.d.contact[: self.d.ncon]:
            if self.floor in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.floor else c.geom1
                bodies.add(self.m.body(self.m.geom_bodyid[other]).name)
        return sorted(bodies)

    def render_wrist(self):
        self.wrist.update_scene(self.d, camera="wrist_cam")
        rgb = self.wrist.render()
        self.wrist.enable_depth_rendering()
        self.wrist.update_scene(self.d, camera="wrist_cam")
        depth = self.wrist.render()
        self.wrist.disable_depth_rendering()
        return rgb, depth

    def render_ext(self):
        self.ext.update_scene(self.d, camera=self.ext_cam, scene_option=self.opt_ext)
        return self.ext.render()

    def project(self, pts_world):
        p, R = self.cam_pose()
        pc = (np.asarray(pts_world) - p) @ R  # world -> camera: R^T (x - p)
        return pc[:, :2] / pc[:, 2:3], pc[:, 2]

    def solve_ik(self, p_goal, R_goal, seeds, iters=500):
        best = None
        for seed in seeds:
            q = np.clip(seed, self.lo, self.hi)
            for _ in range(iters):
                self.set_q(q)
                p, R = self.cam_pose()
                e = np.r_[p_goal - p, rotvec(R_goal @ R.T)]
                if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                    return q, True
                dq = dls(self.jacobian(), e, 0.02)
                n = np.linalg.norm(dq)
                q = np.clip(q + (dq * 0.2 / n if n > 0.2 else dq), self.lo, self.hi)
            score = np.linalg.norm(e[:3]) + 0.05 * np.linalg.norm(e[3:])
            if best is None or score < best[0]:
                best = (score, q)
        return best[1], False


def detect(rgb, min_area=12):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    found = {}
    for name, (_, _, ranges) in MARKERS.items():
        mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in ranges).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        mom = cv2.moments(cnt)
        if mom["m00"] < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if x == 0 or y == 0 or x + w >= rgb.shape[1] or y + h >= rgb.shape[0]:
            continue  # clipped by the image border -> biased centroid
        found[name] = np.array([mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]])
    return found


def interaction_matrix(xy, Z):
    L = np.zeros((2 * len(xy), 6))
    for i, ((x, y), z) in enumerate(zip(xy, Z)):
        L[2 * i] = [-1 / z, 0, x / z, x * y, -(1 + x * x), y]
        L[2 * i + 1] = [0, -1 / z, y / z, 1 + y * y, -x * y, -x]
    return L


BGR = {"red": (40, 40, 230), "green": (40, 200, 40), "blue": (230, 80, 20), "magenta": (220, 40, 220)}

# cv2.waitKeyEx codes for WASD + arrow keys (macOS, Linux/GTK, Windows).
KEYS_UP = {ord("w"), ord("W"), 63232, 65362, 2490368}
KEYS_DOWN = {ord("s"), ord("S"), 63233, 65364, 2621440}
KEYS_LEFT = {ord("a"), ord("A"), 63234, 65361, 2424832}
KEYS_RIGHT = {ord("d"), ord("D"), 63235, 65363, 2555904}
KEY_ESC = 27


def draw_wrist(rgb, uv, uv_star, trails, text):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for i, name in enumerate(MARKERS):
        col = BGR[name]
        if len(trails[i]) > 1:
            cv2.polylines(img, [np.round(np.array(trails[i])).astype(np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(uv_star[i]).astype(int)), 9, col, 2, cv2.LINE_AA)
        if uv is not None:
            cv2.drawMarker(img, tuple(np.round(uv[i]).astype(int)), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv_star[i]).astype(int)), col, 1, cv2.LINE_AA)
    # A dark box instead of an outline: newer OpenCV widens glyphs with thickness.
    box = img[: 14 + 22 * len(text), : 380]
    box[:] = (box * 0.45).astype(np.uint8)
    for j, line in enumerate(text):
        cv2.putText(img, line, (10, 24 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_plot(errors, width, tol, height=160):
    img = np.full((height, width, 3), 250, np.uint8)
    lo, hi = -1.0, 3.0  # log10 px
    to_y = lambda v: int(height - 10 - (np.clip(math.log10(max(v, 1e-3)), lo, hi) - lo) / (hi - lo) * (height - 20))
    for k in range(int(lo), int(hi) + 1):
        y = to_y(10.0**k)
        cv2.line(img, (50, y), (width - 10, y), (215, 215, 215), 1)
        cv2.putText(img, f"1e{k}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(img, (50, to_y(tol)), (width - 10, to_y(tol)), (120, 200, 120), 1)
    if len(errors) > 1:
        xs = np.linspace(50, width - 10, max(len(errors), 300))[: len(errors)]
        pts = np.column_stack([xs, [to_y(e) for e in errors]]).astype(np.int32)
        cv2.polylines(img, [pts], False, (200, 60, 30), 2, cv2.LINE_AA)
    cv2.putText(img, "max feature error [px] (log)", (60, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=here / "so101_6dof_new_calib.urdf")
    ap.add_argument("--out", type=Path, default=here / "visual_servo_results")
    ap.add_argument("--target-xy", type=float, nargs=2, default=[0.24, 0.06], help="Target centre on the table [m]")
    ap.add_argument("--target-yaw-deg", type=float, default=20.0)
    ap.add_argument("--goal-distance", type=float, default=0.15, help="Desired camera-to-target distance [m]")
    ap.add_argument("--goal-pitch-deg", type=float, default=60.0,
                    help="Desired viewing angle below horizontal (90 = straight down; limited by wrist_flex range)")
    ap.add_argument("--goal-yaw-deg", type=float, default=0.0, help="Desired image rotation about the optical axis")
    ap.add_argument("--start-offset", type=float, nargs=3, default=[-0.06, 0.05, 0.02], help="Start camera offset from goal, world [m]")
    ap.add_argument("--start-rot-deg", type=float, nargs=3, default=[12.0, -10.0, 35.0], help="Start camera rotation about camera x/y/z")
    ap.add_argument("--q0-deg", type=float, nargs=6, help="Explicit start joint angles (overrides --start-*)")
    ap.add_argument("--cam-offset", type=float, nargs=2, default=[-0.035, 0.07],
                    help="Camera position in gripper_frame_link: y offset (-y = top of gripper at middle/rest pose), "
                         "distance behind the tip [m]")
    ap.add_argument("--cam-tilt-deg", type=float, default=0.0, help="Camera tilt toward the tool axis")
    ap.add_argument("--fovy", type=float, default=70.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--gain", type=float, default=1.0, help="IBVS gain lambda [1/s]")
    ap.add_argument("--depth", choices=["measured", "desired", "mean"], default="measured",
                    help="Z in the interaction matrix: depth image, goal depth Z*, or 0.5*(L(Z)+L*(Z*))")
    ap.add_argument("--noise-px", type=float, default=0.0, help="Gaussian noise on detected centroids")
    ap.add_argument("--dt", type=float, default=1 / 30)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--tol-px", type=float, default=0.5)
    ap.add_argument("--v-max", type=float, default=0.08, help="Camera linear speed limit [m/s]")
    ap.add_argument("--w-max", type=float, default=0.8, help="Camera angular speed limit [rad/s]")
    ap.add_argument("--qd-max", type=float, default=2.0, help="Joint speed limit [rad/s]")
    ap.add_argument("--damping", type=float, default=0.01, help="DLS damping for the arm Jacobian")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="Show a live window (q/ESC to quit)")
    ap.add_argument("--interactive", action="store_true",
                    help="Move the target with WASD/arrows (Q/E rotate, R reset, ESC quit); the arm keeps servoing")
    ap.add_argument("--target-step", type=float, default=0.005, help="Target move per key press [m]")
    ap.add_argument("--target-yaw-step-deg", type=float, default=5.0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    args.show = args.show or args.interactive
    args.urdf = args.urdf.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    goal_p, goal_R = goal_camera_pose(args.target_xy, args.goal_distance, math.radians(args.goal_pitch_deg),
                                      math.radians(args.goal_yaw_deg))
    sim = Sim(build_model(args, goal_p, goal_R), args.width, args.height)
    names = list(MARKERS)
    pts_w = target_points_world(args.target_xy, math.radians(args.target_yaw_deg))
    pts_w = np.array([pts_w[n] for n in names])

    # Goal: reference joint solution (for validation only; the controller never uses it).
    span = sim.hi - sim.lo
    seeds = [np.zeros(6), np.radians([0, -30, 30, 60, 0, 0])] + [rng.uniform(sim.lo + 0.1 * span, sim.hi - 0.1 * span) for _ in range(20)]
    q_goal, goal_ok = sim.solve_ik(goal_p, goal_R, seeds)
    if not goal_ok:
        print("WARNING: goal camera pose looks unreachable (IK did not converge); IBVS may stall at a joint limit.")
    sim.set_q(q_goal)
    goal_rgb, _ = sim.render_wrist()
    goal_det = detect(goal_rgb)

    # Desired features s*: projection of the known target from the goal camera pose.
    xy_star = (pts_w - goal_p) @ goal_R
    Z_star = xy_star[:, 2].copy()
    xy_star = xy_star[:, :2] / xy_star[:, 2:3]
    uv_star = xy_star * sim.f + sim.c
    goal_det_diff = max((np.linalg.norm(goal_det[n] - uv_star[i]) for i, n in enumerate(names) if n in goal_det), default=float("nan"))

    if args.q0_deg is not None:
        q = np.clip(np.radians(args.q0_deg), sim.lo, sim.hi)
        start_ok = True
    else:
        rx, ry, rz = np.radians(args.start_rot_deg)
        start_R = goal_R @ rot_x(rx) @ rot_y(ry) @ rot_z(rz)
        q, start_ok = sim.solve_ik(goal_p + np.array(args.start_offset), start_R, [q_goal] + seeds)
    sim.set_q(q)
    q_start = q.copy()
    start_rgb, _ = sim.render_wrist()
    if len(detect(start_rgb)) < len(names):
        cv2.imwrite(str(args.out / "start_wrist.png"), cv2.cvtColor(start_rgb, cv2.COLOR_RGB2BGR))
        raise SystemExit(f"Start pose: not all markers visible ({sorted(detect(start_rgb))}). See {args.out / 'start_wrist.png'}")

    mid = np.array([*(np.array(args.target_xy) * 0.5), 0.12])
    sim.ext_cam.lookat[:] = mid
    view_phi = math.atan2(args.target_xy[1], args.target_xy[0])
    side = math.degrees(view_phi) + 90.0  # look across the arm
    sim.ext_cam.distance, sim.ext_cam.azimuth, sim.ext_cam.elevation = 0.7, side, -20.0

    # The goal camera pose is rigidly attached to the target, so s* stays valid when the target moves.
    tgt0_xy, tgt0_yaw = np.array(args.target_xy, dtype=float), math.radians(args.target_yaw_deg)
    tgt_xy, tgt_yaw = tgt0_xy.copy(), tgt0_yaw
    R_t0 = rot_z(tgt0_yaw)
    rel_R, rel_p = R_t0.T @ goal_R, R_t0.T @ (goal_p - np.array([*tgt0_xy, 0.0]))
    view_right = np.array([math.cos(view_phi), math.sin(view_phi)])
    view_up = np.array([-math.sin(view_phi), math.cos(view_phi)])
    target_mocap = sim.m.body("target").mocapid[0]
    ghost_mocap = sim.m.body("goal_ghost").mocapid[0]

    def place_target(xy, yaw):
        nonlocal goal_p, goal_R
        r = np.linalg.norm(xy)
        xy = xy * np.clip(r, 0.16, 0.34) / max(r, 1e-9)
        yaw = tgt0_yaw + np.clip(yaw - tgt0_yaw, -math.pi / 2, math.pi / 2)
        R_t = rot_z(yaw)
        goal_R, goal_p = R_t @ rel_R, np.array([*xy, 0.0]) + R_t @ rel_p
        sim.d.mocap_pos[target_mocap] = [*xy, 0.0]
        sim.d.mocap_quat[target_mocap] = mat2quat(R_t)
        sim.d.mocap_pos[ghost_mocap] = goal_p
        sim.d.mocap_quat[ghost_mocap] = mat2quat(goal_R)
        mujoco.mj_forward(sim.m, sim.d)
        return xy, yaw

    writer = None
    if not args.no_video:
        size = (args.width + 640, max(args.height, 480) + 160)
        for codec in ("avc1", "mp4v"):  # H.264 plays in QuickTime / VS Code; mp4v as fallback
            writer = cv2.VideoWriter(str(args.out / "visual_servo.mp4"), cv2.VideoWriter_fourcc(*codec),
                                     round(1 / args.dt), size)
            if writer.isOpened():
                break
    log_f = open(args.out / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "t", "err_max_px", "err_rms_px", *[f"q_{j}_deg" for j in ARM_JOINTS],
                  "cam_x", "cam_y", "cam_z", "vx", "vy", "vz", "wx", "wy", "wz"])

    trails = [collections.deque(maxlen=300) for _ in names]
    errors, contacts_seen, lost, converged, step = [], set(), 0, False, 0
    for step in itertools.count() if args.interactive else range(args.max_steps + 1):
        t_frame = time.perf_counter()
        rgb, depth = sim.render_wrist()
        det = detect(rgb)
        uv = None
        vc = np.zeros(6)
        converged = False
        if len(det) == len(names):
            lost = 0
            uv = np.array([det[n] for n in names]) + rng.normal(0, args.noise_px, (len(names), 2))
            xy = (uv - sim.c) / sim.f
            e = (xy - xy_star).ravel()
            err_px = np.linalg.norm((uv - uv_star), axis=1)
            errors.append(float(err_px.max()))
            for i in range(len(names)):
                trails[i].append(uv[i])
            if err_px.max() < args.tol_px:
                converged = True
            else:
                if args.depth == "desired":
                    L = interaction_matrix(xy, Z_star)
                else:
                    ij = np.clip(np.round(uv).astype(int), 0, [args.width - 1, args.height - 1])
                    Z = depth[ij[:, 1], ij[:, 0]].astype(float)
                    L = interaction_matrix(xy, Z)
                    if args.depth == "mean":
                        L = 0.5 * (L + interaction_matrix(xy_star, Z_star))
                vc = -args.gain * np.linalg.pinv(L) @ e
                vc[:3] *= min(1.0, args.v_max / max(np.linalg.norm(vc[:3]), 1e-12))
                vc[3:] *= min(1.0, args.w_max / max(np.linalg.norm(vc[3:]), 1e-12))
        else:
            lost += 1

        p_cam, R_cam = sim.cam_pose()
        contacts = sim.floor_contacts()
        contacts_seen.update(contacts)
        log.writerow([step, round(step * args.dt, 4), errors[-1] if uv is not None else "", 
                      float(np.sqrt(np.mean(err_px**2))) if uv is not None else "",
                      *np.degrees(sim.q()).round(4), *p_cam.round(6), *vc.round(6)])

        if writer is not None or args.show:
            text = [f"wrist camera  t={step * args.dt:5.2f}s  step {step}",
                    f"max err {errors[-1]:7.2f} px" if uv is not None else f"markers lost ({sorted(det)})",
                    "circle = desired s*, cross = detected s"]
            if converged:
                text.append("CONVERGED")
            wrist_img = draw_wrist(rgb, uv, uv_star, trails, text)
            ext_img = cv2.cvtColor(sim.render_ext(), cv2.COLOR_RGB2BGR)
            cv2.putText(ext_img, "external view (green box = goal camera pose)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if args.interactive:
                for j, line in enumerate(["WASD/arrows: move target  Q/E: rotate  R: reset  ESC: quit",
                                          f"target x={tgt_xy[0]:.3f} y={tgt_xy[1]:.3f} m  yaw={math.degrees(tgt_yaw):.0f} deg"]):
                    cv2.putText(ext_img, line, (10, 48 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            if contacts:
                cv2.putText(ext_img, "FLOOR CONTACT: " + ",".join(contacts), (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            top = np.zeros((max(args.height, 480), args.width + 640, 3), np.uint8)
            top[: args.height, : args.width] = wrist_img
            top[:480, args.width:] = ext_img
            frame = np.vstack([top, draw_plot(errors[-900:], top.shape[1], args.tol_px)])
            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("SO-101 6DoF visual servo", frame)
                wait_ms = int((args.dt - (time.perf_counter() - t_frame)) * 1000)
                key = cv2.waitKeyEx(max(1, wait_ms))
                if key == KEY_ESC or (not args.interactive and key in (ord("q"), ord("Q"))):
                    break
                if args.interactive and key != -1:
                    step_xy = {**dict.fromkeys(KEYS_UP, view_up), **dict.fromkeys(KEYS_DOWN, -view_up),
                               **dict.fromkeys(KEYS_LEFT, -view_right), **dict.fromkeys(KEYS_RIGHT, view_right)}
                    if key in step_xy:
                        tgt_xy, tgt_yaw = place_target(tgt_xy + step_xy[key] * args.target_step, tgt_yaw)
                    elif key in (ord("q"), ord("Q"), ord("e"), ord("E")):
                        sign = 1 if key in (ord("q"), ord("Q")) else -1
                        tgt_xy, tgt_yaw = place_target(tgt_xy, tgt_yaw + sign * math.radians(args.target_yaw_step_deg))
                    elif key in (ord("r"), ord("R")):
                        tgt_xy, tgt_yaw = place_target(tgt0_xy.copy(), tgt0_yaw)
        if step == 0:
            cv2.imwrite(str(args.out / "start_wrist.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not args.interactive and (converged or lost > 5):
            break

        V = np.r_[R_cam @ vc[:3], R_cam @ vc[3:]]
        dq = dls(sim.jacobian(), V, args.damping)
        dq *= min(1.0, args.qd_max / max(np.abs(dq).max(), 1e-12))
        sim.set_q(np.clip(sim.q() + dq * args.dt, sim.lo, sim.hi))

    log_f.close()
    if writer is not None:
        writer.release()
    final_rgb, _ = sim.render_wrist()
    cv2.imwrite(str(args.out / "final_wrist.png"), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.out / "goal_wrist.png"), cv2.cvtColor(goal_rgb, cv2.COLOR_RGB2BGR))
    if args.show:
        if not args.interactive:
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    p_cam, R_cam = sim.cam_pose()
    q_final = sim.q()
    at_limit = [ARM_JOINTS[j] for j in range(6) if min(q_final[j] - sim.lo[j], sim.hi[j] - q_final[j]) < 1e-3]
    summary = {
        "converged": converged,
        "markers_lost": lost > 5,
        "steps": step,
        "time_s": round(step * args.dt, 3),
        "final_max_error_px": errors[-1] if errors else None,
        "initial_max_error_px": errors[0] if errors else None,
        "final_cam_position_error_mm": float(np.linalg.norm(p_cam - goal_p) * 1000),
        "final_cam_orientation_error_deg": float(np.degrees(np.linalg.norm(rotvec(goal_R @ R_cam.T)))),
        "q_start_deg": np.degrees(q_start).round(3).tolist(),
        "q_final_deg": np.degrees(q_final).round(3).tolist(),
        "q_goal_ik_deg": np.degrees(q_goal).round(3).tolist(),
        "goal_ik_converged": goal_ok,
        "start_ik_converged": start_ok,
        "goal_image_projection_vs_detection_px": float(goal_det_diff),
        "joints_at_limit_final": at_limit,
        "floor_contact_bodies": sorted(contacts_seen),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "scope": "Kinematic simulation (no dynamics, no self-collision check). Camera mount pose is assumed, not from CAD.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, ensure_ascii=False, indent=2))
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
