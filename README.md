# Sim scratchpad

Place this checkout at `IsaacLab/scripts/sim-scratchpad` and use Isaac Lab's
Python environment. Run `git lfs install` before cloning or `git lfs pull` in
an existing checkout to fetch the room scan.
The scan is stored at `assets/3dgs_40k_1m_sh3.usdz`; `demo_env.usd` references it
relative to this checkout.

For development, start with [DEVELOPING.md](DEVELOPING.md): pipeline ownership,
simulated-sand integration points, tests and output locations. Shared robot
geometry lives in [assets/urdf/](assets/urdf/README.md); generated experiments
belong in [logs/](logs/README.md).

- Open `demo_env.usd` in Isaac Sim for the registered room splat and calibrated
  cabin and arm cameras. [IR capture, replay, and calibration](camera_calibration/README.md)
- Run `uv run --no-sync python sensor_demo/demo.py --view soil` for the Newton
  soil demo. Its ROS launcher publishes the ground-truth heightmap, and
  `sensor_demo/collect.sh` records expert episodes through `ros-excavator`.
  [Newton demo and data collection](sensor_demo/README.md)
- Both RTX lidar mounts are already in `demo_env.usd`. The [scan reader](ros2/sim_lidar.py)
  can be used from Kit, and the [lidar prototype](lidar_demo_script.py) shows how
  to filter robot self-hits.

## 3DGS viewer

Run from this checkout using Isaac Lab's existing environment, including its
`isaacsim` extra. On this Windows setup, Isaac Sim 6.2 and the room scan are
already installed; `--no-sync` reuses that environment.
The launcher shares the desktop application's versioned shader caches under
`%LOCALAPPDATA%/ov/cache/shaders` instead of rebuilding them in Kit's portable cache.

```powershell
uv run --no-sync python launch_3dgs.py
uv run --no-sync python launch_3dgs.py --camera cabin
uv run --no-sync python launch_3dgs.py --camera arm
```

The viewer opens `demo_env.usd` with the registered Gaussian splat, excavator,
and calibrated cameras. Physics starts paused. Use the viewport camera menu to
switch views and close the window to exit. Camera changes stay in the session
layer unless you explicitly save the scene.

Save a viewport image, or render a fixed number of frames without a window:

```powershell
uv run --no-sync python launch_3dgs.py --snapshot logs/3dgs/overview.png
uv run --no-sync python launch_3dgs.py --headless --frames 120 --camera cabin --snapshot logs/3dgs/cabin.png
```

Run the camera/map demo with recorded motion, or drive it with a gamepad:

```powershell
uv run --no-sync python launch_3dgs.py --mode replay
uv run --no-sync python launch_3dgs.py --mode gamepad
uv run --no-sync python launch_3dgs.py --mode gamepad --nodynamics
```

This uses the checked-in [camera/filter profile](config/heightmaps/README.md) and
[motion sample](assets/motions/README.md). No recorded dataset or files from
`logs/` are needed. It shows cabin/boom IR views at 224x168 (the original 4:3 field
of view), a 64x64 front map and a 160x160 surrounding FastDEM map. Viewport 1 is
disabled; full 640x480 depth sensors still feed mapping in the background.
Open `demo_env.usd` in Isaac Sim to inspect the full environment, or use
`--show-viewport` for a performance comparison. `--ir-width` changes the IR render size.
Use `--depth-maps` instead of `--mode replay`
for a stationary scene, or `--replay PATH` for another saved motion.

Gamepad mode loads the arm and slew MLPs from the sibling `Isaac-hydraulic-actuator`
checkout; set `--actuator-repo PATH` or `HYDRAULIC_ACTUATOR_REPO` elsewhere. The
arm model includes carriage pitch. Left stick X/Y controls slew/tilt, right stick
Y/X controls lift/bucket, and B or keyboard R resets pose, model history and maps.
`--nodynamics` uses direct joint speeds. Replay always reproduces saved poses exactly.

The models run on a separate CPU worker with fixed 10 ms steps. Rendering cannot
stretch their history timestep; long scheduling stalls are reported as dropped
wall time. Motion is integrated into USD for free-space operation; contact and
digging loads are not modeled. This gamepad path is currently non-ROS.
Control rates, inference time, clipping and rendering timings are saved to
`control_performance.json` under `--depth-out`. Rendering still limits visual
feedback and gamepad polling, even when control maintains 100 Hz.

The shared mapper still requires a `kaivuriprokkis` checkout (`--robot-repo PATH`
or `KAIVURI_REPO`) and a locally built FastDEM adapter. See
[setup on another computer](DEVELOPING.md#run-on-another-computer).
`--config PATH` selects another simulator profile; `--dataset PATH` explicitly
uses recorded calibration and the robot checkout's current policy for comparisons.
[Calibration, launch commands, exports and filter experiments](DEPTH_MAPPING.md).

Publish processed heightmaps and FastDEM terrain for remote RViz with `--ros`.
Raw depth and grayscale cameras are opt-in. [ROS launch and Ubuntu handoff](ros2/LATEST.md).

The excavator USD comes from
[Isaac-hydraulic-actuator at 4775a97](https://github.com/eetmie/Isaac-hydraulic-actuator/tree/4775a97)
and retains its [Apache 2.0 license](assets/excavator_bucket.LICENSE). The NVIDIA
SimReady objects retain their CC BY 4.0 license and attribution metadata in
`assets/simready/`.
