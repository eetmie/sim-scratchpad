# Excavator sim -> ROS 2 -> RViz

Isaac Sim (Kit, Python 3.12) uses its **bundled ROS 2 Jazzy** libs; the system has **Lyrical** (Python 3.14),
which Kit can't load. They talk over DDS with Fast DDS on both sides (tested: JointState, Image, PointCloud2).

## Run

```bash
# 1. once (or after changing the robot): URDF + meshes -> ros2/urdf/
uv run --extra isaacsim python scripts/sim-scratchpad/ros2/export_urdf.py

# 2. Isaac Sim with the ROS env (do not source /opt/ros/lyrical in this shell)
scripts/sim-scratchpad/ros2/run_isaacsim.sh --exec scripts/sim-scratchpad/ros2/open_demo.py
#    opens demo_env.usd, runs ros2_bridge.py (it also starts the teleop + lidar) and presses Play.
#    Without --exec: open demo_env.usd, run ros2_bridge.py in the Script Editor, press Play.

# 3. another shell: RViz + robot_state_publisher
source /opt/ros/lyrical/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch scripts/sim-scratchpad/ros2/view_excavator.launch.py
```

## Topics (sim time; fixed frame `world`)

| Topic | Type | Rate | Notes |
| --- | --- | --- | --- |
| `/clock` | `rosgraph_msgs/Clock` | every frame | resets to 0 on Stop |
| `/joint_states` | `sensor_msgs/JointState` | every frame | rad; `revolute_carriage, revolute_lift, revolute_tilt, revolute_tool` |
| `/cam_cabin/image_raw`, `/cam_boom/image_raw` | `sensor_msgs/Image` rgb8 | every frame | 320 x 240 (`CAMERA_RES`), plus `camera_info` |
| `/lidar_cabin/points`, `/lidar_boom/points` | `sensor_msgs/PointCloud2` xyz | ~5 Hz | frame `world`, excavator hits removed |
| `/excavator/heightmap` | `sensor_msgs/Image` 32FC1 | ~5 Hz | 64 x 64, m above ground, NaN = unseen; row 0 = +X edge, col 0 = +Y edge; X 0.16..0.80, Y -0.32..0.32 m |
| `/excavator/heightmap_cloud` | `sensor_msgs/PointCloud2` xyz | ~5 Hz | known cell centers, for the 3D view |

`/tf` comes from `robot_state_publisher` (URDF + `/joint_states`), as it would on the real robot.
Camera frames (`cam_cabin`, `cam_boom`) are not in the URDF yet; the Image panels don't need them.

## Files

- `export_urdf.py`: USD -> URDF with Isaac Sim's exporter, then fixes its frames: joint origins/axes come from the
  USD joint frames (so URDF angles == `/joint_states`), link frames are the USD body frames, and each mesh gets
  the exact origin that puts it where the USD has it. Plus a `world` root link.
- `check_urdf_fk.py`: places every URDF visual mesh at the sim's joint angles and compares it with the sim's mesh
  (max vertex error per link). Run after exporting.
- `ros2_bridge.py`: Script Editor helper; OmniGraph for clock/joints/cameras, rclpy for lidar + height map.
  `SHOW_LIDAR_POINTS_IN_SIM = False` hides the lidar debug points, which would otherwise appear in the camera images.
- `open_demo.py`: Kit startup script for `--exec` (open stage, start bridge, Play).
- `run_isaacsim.sh`: launches Isaac Sim with the bundled Jazzy env.
- `view_excavator.launch.py`, `excavator.rviz`: RViz side.
- `smoke_test.sh` (+ `headless_run.py`): headless sim for 60 s, checks every topic arrives in Lyrical
  and that the bridge graph stays out of `demo_env.usd` (it lives on the session layer). Run after changes.

Not done yet: joint commands in (`-1..1` rate commands), camera frames in TF.
