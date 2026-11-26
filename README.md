# Sim scratchpad

Place this checkout at `IsaacLab/scripts/sim-scratchpad` and use Isaac Lab's
Python environment. Run `git lfs install` before cloning or `git lfs pull` in
an existing checkout to fetch the room scan.

- Open `demo_env.usd` in Isaac Sim for the registered room splat and calibrated
  cabin and arm cameras. [IR capture, replay, and calibration](camera_calibration/README.md)
- Run `uv run --no-sync python sensor_demo/demo.py --view soil` for the Newton
  soil demo. Its ROS launcher publishes the ground-truth heightmap, and
  `sensor_demo/collect.sh` records expert episodes through `ros-excavator`.
  [Newton demo and data collection](sensor_demo/README.md)
- Both RTX lidar mounts are already in `demo_env.usd`. The [scan reader](ros2/sim_lidar.py)
  can be used from Kit, and the [lidar prototype](lidar_demo_script.py) shows how
  to filter robot self-hits.

The excavator USD comes from
[Isaac-hydraulic-actuator at 4775a97](https://github.com/eetmie/Isaac-hydraulic-actuator/tree/4775a97)
and retains its [Apache 2.0 license](assets/excavator_bucket.LICENSE). The NVIDIA
SimReady objects retain their CC BY 4.0 license and attribution metadata in
`assets/simready/`.
