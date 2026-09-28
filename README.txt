SO-101 6DoF URDF package

Files:
- so101_6dof_new_calib.urdf
- assets/wrist_pitech_yaw_joint_2_fixed.stl  (Fusion004)
- assets/wrist_roll.stl                      (Fillet001)

Also copy the standard SO-101 assets from TheRobotStudio/SO-ARM100
Simulation/SO101/assets into this assets/ directory.

Motor mapping:
1 shoulder_pan
2 shoulder_lift
3 elbow_flex
4 wrist_flex
5 wrist_yaw
6 wrist_roll
7 gripper

Important:
- wrist_yaw limit is currently provisional +/-90 degrees.
- Inertial parameters for the two modified wrist links are intentionally omitted.
  Recalculate mass/inertia before dynamics/physics simulation.
- The original SO-101 gripper geometry is retained. The camera-mount mesh embedded
  in the supplied FCStd is not referenced because no standalone STL for it was supplied.
