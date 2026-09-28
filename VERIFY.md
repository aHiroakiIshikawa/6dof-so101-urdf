SO-101 6DoF URDF verification

1. Put the standard SO-101 mesh assets in:
   ./assets/

2. Install/use LeRobot with the PlaCo dependency:
   uv pip install -e ".[placo-dep]"

3. Numerical FK test:
   uv run python verify_so101_6dof.py --urdf ./so101_6dof_new_calib.urdf

4. MeshCat visual verification:
   uv run python verify_so101_6dof.py \
     --urdf ./so101_6dof_new_calib.urdf \
     --visualize

The viewer steps through:
  zero
  wrist_yaw +30 deg
  wrist_yaw -30 deg
  wrist_roll +30 deg
  wrist_roll -30 deg

Inspect these frames:
  wrist_link
  wrist_yaw_link
  wrist_roll_link
  gripper_link
  gripper_frame_link

Current expected kinematic convention:
  motor ID4: wrist_flex
  motor ID5: wrist_yaw
  motor ID6: wrist_roll
  motor ID7: gripper

Do not command the physical robot from this verification script.
