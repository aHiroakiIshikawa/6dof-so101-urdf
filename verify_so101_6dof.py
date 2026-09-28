#!/usr/bin/env python3
"""
SO-101 6DoF URDF verification utility.

Checks:
  1. Load the URDF with LeRobot RobotKinematics (PlaCo backend).
  2. Print FK for zero, +/- yaw, +/- roll.
  3. Optionally open the PlaCo/MeshCat viewer and step through test poses.

Usage:
  uv run python verify_so101_6dof.py \
      --urdf ./so101_6dof_new_calib.urdf

  uv run python verify_so101_6dof.py \
      --urdf ./so101_6dof_new_calib.urdf \
      --visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
]

TEST_POSES_DEG = {
    "zero": np.array([0, 0, 0, 0, 0, 0], dtype=float),
    "yaw +30": np.array([0, 0, 0, 0, 30, 0], dtype=float),
    "yaw -30": np.array([0, 0, 0, 0, -30, 0], dtype=float),
    "roll +30": np.array([0, 0, 0, 0, 0, 30], dtype=float),
    "roll -30": np.array([0, 0, 0, 0, 0, -30], dtype=float),
}


def print_pose(name: str, T: np.ndarray) -> None:
    R = T[:3, :3]
    p = T[:3, 3]
    print(f"\n[{name}]")
    print("position [m] =", np.array2string(p, precision=6, suppress_small=True))
    print("frame X in base =", np.array2string(R[:, 0], precision=6, suppress_small=True))
    print("frame Y in base =", np.array2string(R[:, 1], precision=6, suppress_small=True))
    print("frame Z in base =", np.array2string(R[:, 2], precision=6, suppress_small=True))
    print("T_base_gripper_frame =")
    print(np.array2string(T, precision=6, suppress_small=True))


def check_mesh_files(urdf_path: Path) -> list[Path]:
    """Return missing relative STL/mesh files referenced by the URDF."""
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    missing: list[Path] = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith(("package://", "http://", "https://")):
            continue
        path = (urdf_path.parent / filename).resolve()
        if not path.exists():
            missing.append(path)
    return sorted(set(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--target-frame",
        default="gripper_frame_link",
        help="URDF frame used as the end effector.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open PlaCo/MeshCat and step through zero/yaw/roll test poses.",
    )
    args = parser.parse_args()

    urdf = args.urdf.expanduser().resolve()
    if not urdf.exists():
        print(f"ERROR: URDF not found: {urdf}", file=sys.stderr)
        return 2

    missing = check_mesh_files(urdf)
    if missing:
        print("\nWARNING: The following visual meshes are missing:")
        for path in missing:
            print("  ", path)
        print(
            "\nNumerical kinematics may still work depending on the PlaCo build, "
            "but full MeshCat visualization requires all referenced meshes."
        )

    try:
        from lerobot.model.kinematics import RobotKinematics
    except Exception as exc:
        print(
            "\nERROR: LeRobot could not be imported.\n"
            "Run this from your LeRobot environment.",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 3

    try:
        kin = RobotKinematics(
            urdf_path=str(urdf),
            target_frame_name=args.target_frame,
            joint_names=ARM_JOINTS,
        )
    except Exception as exc:
        print(
            "\nERROR: RobotKinematics failed to load the URDF.\n"
            "Check the details below. Missing mesh files, an invalid URDF, or an "
            'unavailable "placo-dep" dependency can each prevent loading.',
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 4

    print("Loaded URDF with LeRobot RobotKinematics")
    print("URDF:", urdf)
    print("Target frame:", args.target_frame)
    print("Arm joints:", ARM_JOINTS)

    poses = {}
    for name, q_deg in TEST_POSES_DEG.items():
        T = kin.forward_kinematics(q_deg)
        poses[name] = T
        print_pose(name, T)

    # Direction summary relative to zero pose.
    zero = poses["zero"]
    for name in ("yaw +30", "yaw -30", "roll +30", "roll -30"):
        dp = poses[name][:3, 3] - zero[:3, 3]
        print(
            f"\n{name:>8} EE displacement from zero [mm] = "
            f"{np.array2string(dp * 1000.0, precision=3, suppress_small=True)}"
        )

    if not args.visualize:
        print(
            "\nNumerical FK check finished.\n"
            "Add --visualize to inspect the actual URDF geometry and coordinate frames in MeshCat."
        )
        return 0

    if missing:
        print(
            "\nERROR: --visualize requested but mesh files are missing. "
            "Copy the standard SO-101 assets into the URDF's assets/ directory first.",
            file=sys.stderr,
        )
        return 5

    try:
        from placo_utils.visualization import robot_frame_viz, robot_viz
    except Exception as exc:
        print(
            "\nERROR: placo_utils visualization is unavailable.",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 6

    robot = kin.robot
    viz = robot_viz(robot)

    frames = [
        "wrist_link",
        "wrist_yaw_link",
        "wrist_roll_link",
        "gripper_link",
        args.target_frame,
    ]

    print(
        "\nMeshCat viewer initialized. Open the URL printed by MeshCat if a browser "
        "window did not open automatically."
    )
    print("Coordinate frames to inspect:", frames)
    print(
        "\nInterpretation:\n"
        "  + wrist_yaw must rotate the downstream wrist by the right-hand rule\n"
        "    about the displayed wrist_yaw axis.\n"
        "  + wrist_roll must rotate only around the roll axis after yaw.\n"
        "  + At zero, compare gripper orientation with the real robot before\n"
        "    accepting gripper_mount_joint."
    )

    for name, q_deg in TEST_POSES_DEG.items():
        for joint_name, deg in zip(ARM_JOINTS, q_deg):
            robot.set_joint(joint_name, np.deg2rad(deg))
        robot.update_kinematics()

        for frame in frames:
            try:
                robot_frame_viz(robot, frame, scale=0.06)
            except Exception as exc:
                print(f"WARNING: could not display frame {frame}: {exc}")

        viz.display(robot.state.q)

        print(f"\nShowing pose: {name}  q_deg={q_deg.tolist()}")
        try:
            input("Press Enter for the next pose (Ctrl-C to quit)...")
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0

    print("\nAll test poses displayed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
