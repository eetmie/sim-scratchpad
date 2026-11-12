# Excavator sim -> ROS 2

The sim plays the robot: Isaac Sim drives the joints from `/excavator/command` and publishes what the robot's
sensors would (two D435i-like cameras: RGB + noisy depth). `demo_env.usd` holds the excavator, the two cameras
(`cam_cabin`, `cam_boom`) and two lidars (`lidar_cabin`, `lidar_boom`; optional, not used by the map). Everything else is ordinary ROS nodes that run the same
on the real robot: gamepad teleop, the depth self-filter, FastDEM height mapping, the VLA height image and the
normalized joint angles. `sensors.py` holds the shared contract (topics, frames, camera mounts, joint bounds).

Isaac Sim (Kit, Python 3.12) uses its **bundled ROS 2 Jazzy** libs; the system has **Lyrical** (Python 3.14),
which Kit can't load. They talk over DDS with Fast DDS on both sides.

## Run

```bash
# Once (with Lyrical sourced), and after changing depth_to_cloud_cpp/: FastDEM + depth_to_cloud into ros2/ws
scripts/sim-scratchpad/ros2/setup_ws.sh

# Terminal 1 - the sim. Don't source /opt/ros/lyrical here.
cd ~/IsaacLab
scripts/sim-scratchpad/ros2/run_isaacsim.sh --exec scripts/sim-scratchpad/ros2/open_demo.py

# Terminal 2 - ROS side: TF, gamepad teleop, self-filter + FastDEM, VLA height image, RViz
cd ~/IsaacLab
source /opt/ros/lyrical/setup.bash
source scripts/sim-scratchpad/ros2/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch scripts/sim-scratchpad/ros2/view_excavator.launch.py   # teleop/mapping/rviz:=false to leave out
```

- Gamepad (ISO layout, any SDL pad): left stick X swing, left stick Y arm, right stick X bucket, right stick Y boom.
  Swing has no end stop, like the real machine; the other joints stop inside the robot's joint bounds.
- Clear the map: `ros2 service call /fastdem/reset_map std_srvs/srv/Trigger` (Stop in Isaac clears it too).
- Record VLA data: `ros2 bag record /excavator/command /excavator/joint_angles /excavator/heightmap/image
  /excavator/cameras/cabin/image_raw /excavator/cameras/arm/image_raw`.
- The Isaac viewport window is hidden, so it doesn't render (RViz is the view); that saves a full RTX render per
  frame. Window > Viewport > Viewport 1 brings it back. Options for terminal 1: `EXCAVATOR_VIEWPORT=1` keeps it; `EXCAVATOR_LIDARS=1` adds the two RTX
  lidars (`sim_lidar.py`); `EXCAVATOR_PROFILE=1` prints the app frame rate and the bridge's ms per step every 10 s
  (`depth_to_cloud` always logs its own).
- Rates are sim time; when the sim runs below real time, the wall-clock rates drop with it.
- After changing the robot in `demo_env.usd`: `uv run --extra isaacsim python scripts/sim-scratchpad/ros2/export_urdf.py`.

## Pipeline

```
Isaac Sim (ros2_bridge.py)                 ROS (same on the robot)
  depth 16UC1 + camera_info, 5 Hz ------> depth_to_cloud (C++) ----> FastDEM ---+--> /fastdem/mapping/cloud
  RGB, 30 Hz                               stride 2, max 0.9 m,                  |    remembered map (RViz)
  /joint_states --> robot_state_publisher  URDF self-filter at the image stamp,  |
  /excavator/command <-- teleop_joy        optional depth correction             +--> heightmap_image
                                                                                      /excavator/heightmap(/image)
                                                                                      latest scan, cabin frame (VLA)
```

- **Frames.** `cabin_camera` / `arm_camera` are optical depth (IR1) frames. The cabin mount is the real
  calibration (`sand-calibration.json` via `kaivuri_bridge.camera_transforms`). The arm mount is a placeholder (`cam_boom`'s
  pose): the real arm calibration puts the camera 0.74 m off the arm (see `sensors.py`). The bridge re-applies pose
  and lens from `sensors.py` to the USD cameras at startup, so `sensors.py` stays the source of truth. The sim
  publishes sensor frames from `view_excavator.launch.py`; the robot publishes its calibrated ones itself.
- **Self-filter.** One oriented box per URDF link mesh (`robot_boxes.py`, computed at launch), padded 25 mm,
  placed with TF at the image stamp. At rest the arm camera drops ~4.5k bucket points per image; ~5 flying-pixel
  edge points get through.
- **Depth correction** (robot): `mapping.launch.py calibration:=~/depth-live-bench/sand-calibration.json` loads
  `sand_height.py`'s stereo "dome" fit (`depth_correction`, z' = z / (1 - z k(u, v))); off without it.
  `depth_correction_check.py` checks the C++ against `sand_height.py`'s formula.
- **FastDEM** (`fastdem.yaml`): 160 x 160 cells of 1 cm, world-aligned, centred on the swing axis.
  `base_frame` is `upper_carriage` because FastDEM caches sensor->base as a static transform; the swing comes in
  through the per-scan pose. `setup_ws.sh` builds it at a pinned commit with `fastdem.patch` (fixes from
  `FastDEM_Update_Notes.md`: map stamps = scan time, out-of-order scans dropped, clock resets clear the map,
  TF buffers survive a Stop in Isaac, `std_srvs` dependency, nanoGrid pinned, builds on Lyrical).
- **VLA height image** (`heightmap_image.py`): FastDEM's latest scan only, top-down in `upper_carriage`, 64 x 64
  over x 0.1..0.9 m, y -0.4..0.4 m, so it turns with the cabin like the camera. 32FC1 in m, plus mono8 (0 = unseen).
- **Sim noise.** Passive stereo (range-squared, holes, edge flying pixels, 0.17 m min-Z) in `ros2_bridge.py`;
  tune `STEREO_SUBPIXEL` against a real recording. The sim has no stereo dome.

## Topics (sim time; fixed frame `world`)

| Topic | Type | Rate | Notes |
| --- | --- | --- | --- |
| `/excavator/command` (in) | `std_msgs/Float32MultiArray` | 50 Hz | `[slew, lift, tilt, scoop]` valve demand -1..1; stale after 150 ms = stop |
| `/clock` | `rosgraph_msgs/Clock` | every frame | resets to 0 on Stop |
| `/joint_states` | `sensor_msgs/JointState` | every frame | rad; `revolute_carriage, revolute_lift, revolute_tilt, revolute_tool` |
| `/excavator/joint_angles` | `std_msgs/Float32MultiArray` | every frame | `[slew, lift, tilt, scoop]` in -1..1, the robot's normalization |
| `/excavator/cameras/{cabin,arm}/image_raw` | `sensor_msgs/Image` rgb8 | 30 Hz | 640 x 480 (the robot's bridge still sends IR mono8) |
| `/excavator/cameras/{cabin,arm}/depth/image_raw` | `sensor_msgs/Image` 16UC1 | 5 Hz | mm, 0 = no depth, frame `{cabin,arm}_camera` |
| `/excavator/cameras/{cabin,arm}/depth/camera_info` | `sensor_msgs/CameraInfo` | 5 Hz | fx from the real cameras, centred principal point |
| `/excavator/cameras/cabin/points` | `sensor_msgs/PointCloud2` | 5 Hz | self-filtered, camera frame (`.../points_self`: dropped, when subscribed) |
| `/fastdem/mapping/cloud`, `/fastdem/mapping/gridmap` | PointCloud2, `grid_map_msgs/GridMap` | 5 Hz | remembered map |
| `/excavator/heightmap`, `/excavator/heightmap/image` | `sensor_msgs/Image` 32FC1, mono8 | 5 Hz | VLA height image |
| `/lidar_cabin/points`, `/lidar_boom/points` | `sensor_msgs/PointCloud2` | ~5 Hz | with `EXCAVATOR_LIDARS=1`; lidar frame, excavator hits included |

## Files

- `sensors.py`: the sim/robot contract: camera mounts and intrinsics, lidar frames, topics, joint bounds, commands.
- `ros2_bridge.py`: Kit side; OmniGraph for `/clock` + `/joint_states`, rclpy for `/excavator/command` (joint
  drives), cameras (with stereo noise) and lidars. `sim_lidar.py`: the optional RTX lidars.
- `open_demo.py`: Kit startup script for `--exec` (open stage, start bridge, Play). `run_isaacsim.sh`: Isaac Sim
  with the bundled Jazzy env.
- `depth_to_cloud_cpp/`: C++ node: depth image -> self-filtered cloud (+ depth correction). `robot_boxes.py`: its
  link boxes from the URDF meshes.
- `mapping.launch.py`, `fastdem.yaml`: depth_to_cloud + FastDEM + heightmap_image. `heightmap_image.py`: the VLA
  height image.
- `teleop_joy.py`, `teleop.launch.py`: gamepad (joy `game_controller_node`) -> `/excavator/command`.
- `joint_angles.py`: sim only; `/joint_states` -> `/excavator/joint_angles` like the robot's bridge.
- `view_excavator.launch.py`, `excavator.rviz`: everything ROS-side for the sim (robot_state_publisher, sensor
  frames, teleop, mapping, joint_angles, RViz).
- `setup_ws.sh`, `fastdem.patch`: builds `ros2/ws` (FastDEM + grid_map_msgs at pinned commits, depth_to_cloud).
- `export_urdf.py`: USD -> URDF with Isaac Sim's exporter, then fixes its frames so URDF angles == `/joint_states`
  and every mesh sits where the USD has it. `check_urdf_fk.py`: compares URDF meshes with the sim's. Run after export.
- Checks: `smoke_test.sh [--lidars]` (+ `headless_run.py`, `command_check.py`): headless sim for 90 s with the ROS
  launch; every topic arrives, commands drive the arm, FastDEM maps carry the scan time, the bridge graph stays out
  of `demo_env.usd`. `depth_correction_check.py`: no sim needed. Run after changes.

## Not done yet

- Arm camera into the map: FastDEM caches sensor->base as static, so the boom camera needs a per-stamp extrinsic in
  FastDEM's `tf_bridge.hpp` first. The arm calibration itself is off by 0.74 m (above).
- The robot side: publish depth `image_raw` + `camera_info` from the native stack, and `/excavator/command` signs
  still need commissioning (`COMMAND_SIGN` in `sensors.py`).
- FastDEM: variance/obstacle layer semantics, robust low-outlier handling, Jetson benchmark with the VLA running
  (`FastDEM_Update_Notes.md`).
- Performance: rendering the two 640 x 480 cameras is the biggest cost, then the depth noise model (~12 ms/camera).
