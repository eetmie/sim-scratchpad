# Depth self-filter + FastDEM (ROS side)

Restored from the robot's copy after the 8446cb3 cleanup. End use is over ROS; the same launch runs on the robot.

```bash
source /opt/ros/<distro>/setup.bash && scripts/sim-scratchpad/ros2/setup_ws.sh   # FastDEM @5cd3e25 + fastdem.patch, depth_to_cloud
source scripts/sim-scratchpad/ros2/ws/install/setup.bash
ros2 launch scripts/sim-scratchpad/ros2/mapping.launch.py               # sim time; use_sim_time:=false on the robot
```

- `depth_to_cloud_cpp/`: depth image -> cloud with URDF link boxes removed at the image stamp (`robot_boxes.py`).
- `fastdem.patch`: scan-time map stamps, out-of-order / clock-reset handling, steady-clock TF buffer, pinned nanoGrid.
  Its `fastdem/` core part is kaivuriprokkis `native_stack/tools/heightmap_bench/fastdem-core.patch`.
- `fastdem.yaml` is identical to kaivuriprokkis `heightmap_bench/fastdem.yaml`.

The robot's tuned arm filter (FK arm-mesh distance grid + margins, arm-camera view/near cuts) lives in kaivuriprokkis
`heightmap_bench/mapper.py` with settings in `native_stack/config/heightmap_pipeline.json`. To try filters offline on
recorded robot data, see "Offline from a complete export" in `heightmap_bench/README.md`.
