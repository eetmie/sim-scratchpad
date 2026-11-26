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

`depth_to_cloud` runs the robot's arm filter (kaivuriprokkis `heightmap_bench/mapper.py`, values from its
`config/heightmap_pipeline.json`, set at the top of `mapping.launch.py`):

- boom/arm/bucket: `arm_model.npz` distance grids (copied from kaivuriprokkis `config/arm_model.npz`, built from this
  URDF by its `tools/arm_model.py`), margins 0.09 / 0.045 / 0.10 m; `arm_grid.py` places them on the URDF links and
  refuses an npz whose link lengths do not match this URDF. Other links (upper carriage) keep their padded boxes.
- silhouette: pixels within 8 px of the projected arm model (the robot's 2 cells at stride 4).
- camera bodies: points within 0.09 m of any camera frame.
- arm camera only: skip frames looking back past -5 deg; drop points nearer than 0.24 m in front of the slew axis.

Not ported (mapping, not filtering): the robot's 0.14-0.80 m annulus and |z| <= 0.30 m crop, and cabin-first fusion
with the arm camera (FastDEM maps the cabin camera only). The Newton sim does not publish depth yet, so this
runs on the robot or a depth-publishing sim.

Checks: `python3 ros2/arm_grid.py` (grids sit on the URDF arm surface) and, with the workspace sourced,
`python3 ros2/test_arm_filter.py` (synthetic cabin depth at three arm poses: all arm pixels cut, ground clear of the
margins kept). To try filter settings on recorded robot data, see "Offline from a complete export" in kaivuriprokkis
`heightmap_bench/README.md`.
