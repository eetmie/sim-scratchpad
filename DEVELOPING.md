# Developing the simulator

This is an independent repository inside Isaac Lab. Use the enclosing Isaac Lab
Python environment (`uv run --no-sync`) and run commands from this checkout.
Fetch LFS assets before starting the viewer. The main heightmap entry point is
`launch_3dgs.py --mode replay` (motion plus depth/maps) or `--depth-maps` (stationary).
`--mode gamepad` drives the camera/map demo with the external arm/slew MLPs;
`--nodynamics` selects direct joint speeds. The Newton soil demo has its own entry point.

## Find the code

| Area | Source and responsibility |
| --- | --- |
| Viewer | `launch_3dgs.py`: app lifecycle, replay, capture scheduling, optional ROS |
| Simulated depth | `depth_mapping.py`: calibrated cameras, render/readback, maps and exports |
| Robot mapping | `robot_mapping.py`: bridge to the external robot policy and FastDEM |
| Simulator settings | `config/heightmaps/`: camera calibration, filter policy, ground reference and FastDEM settings |
| Demo motion | `assets/motions/excavator_demo.json`, played by `scene_motion.py` |
| Gamepad motion | `hydraulic_motion.py`: USD joint-frame FK, external MLP loading and fixed 100 Hz worker |
| Compact IR views | `ir_views.py`: scaled camera calibration and grayscale render products |
| Calibration | `camera_calibration/`: camera mounts, recorded poses and calibration tools |
| Offline comparisons | `replay_depth.py`, `offline_robot_maps.py`, `benchmark_robot_maps.py` |
| Simulated soil | `sensor_demo/soil.py`: soil state; `surface.py`: rendered surface; `demo.py`: simulation loop |
| Shared viewer helpers | `newton/`: input, cameras and mesh publishing; `sand.py` is the older lightweight sand model |
| ROS transport | `ros2/viewer_bridge.py`, `viewer_scene.py`, `map_contract.py`; see [LATEST.md](ros2/LATEST.md) |
| Shared assets | Room scan, robot USD/URDF, arm mask, demo motion and SimReady objects under `assets/` |
| Tests | `tests/`: heightmap/replay/ROS contracts; subsystem tests stay beside calibration and soil code |
| Inspection tools | `tools/`: reusable utilities with command-line arguments |
| Generated output | `logs/<workflow>/<run>/`; soil recordings currently use `sensor_demo/runs/` |
| Build cache | `.cache/fastdem/`: compiler, third-party sources, licenses and build products |

Keep source and tests out of ignored output directories. Use explicit run names
when comparing changes, such as `logs/depth_mapping/sand-baseline` and
`logs/depth_mapping/sand-updated`. Existing command-line entry points stay at the
repository root so launch scripts and imports remain stable.

## Heightmap pipeline

```text
demo_env.usd + calibrated cabin/arm cameras
    -> DepthMapping: rendered optical-Z depth + live camera/link poses
    -> RobotPolicy: projection, robot masks, cabin-first fusion
    -> FastDEM: accumulated surrounding terrain
    -> latest.npz / viewer panels / optional ViewerRosBridge
```

The `kaivuriprokkis` checkout owns the mapping algorithms. Set `KAIVURI_REPO` or
pass `--robot-repo`; do not copy its filter implementations into the simulator.
The simulator owns its validated settings in `config/heightmaps/`, including the
camera/map profile and filter policy. `--config PATH` selects another profile.
No dataset is needed for the viewer or the bundled motion demo.

`--dataset PATH` explicitly selects recorded camera/map calibration and the robot
checkout's current filter settings for comparisons. It is mutually exclusive
with `--config`, and remains required by the offline recorded-data tools.

## Run on another computer

Use the same Isaac Lab checkout/version with Python 3.12 and Isaac Sim 6.2.
Initialize that environment with `uv sync --extra isaacsim` from its root if
needed. Clone this repository under `IsaacLab/scripts/sim-scratchpad`, run
`git lfs pull` there, and clone the matching `kaivuriprokkis` revision (the
profile records its baseline revision). No `heightmap_tests_complete` copy or
old `logs/` directory is required. Run the following from sim-scratchpad.

Build FastDEM once on each development machine:

```powershell
uv run --no-sync --with cmake --with ninja python build_fastdem.py --llvm-mingw --robot-repo C:/path/to/kaivuriprokkis
```

On Linux omit `--llvm-mingw` and have a C++ toolchain installed.
New builds use `.cache/fastdem/`; the runtime falls
back to the existing `logs/fastdem/` build on older checkouts. Do not move a
configured CMake tree: its cache contains absolute paths. `--work` changes the
build location and `--fastdem` selects a particular executable.

```powershell
uv run --no-sync python launch_3dgs.py --mode replay --robot-repo C:/path/to/kaivuriprokkis --depth-out logs/depth_mapping/sand-baseline
uv run --no-sync python tools/inspect_heightmaps.py logs/depth_mapping/sand-baseline --plot
```

Use the viewer's save button or close it to export a capture. Detailed formats,
replay commands and experiments are in [DEPTH_MAPPING.md](DEPTH_MAPPING.md).
Use `--depth-maps` instead of `--mode replay` for a stationary scene. The motion sample
is about 99 KiB and only animates poses; it does not drive hydraulic physics.

## Adding simulated sand

The calibrated room viewer currently observes a static Gaussian scan. The
Newton soil simulation in `sensor_demo/` is a separate demo; it is not yet wired
into that viewer. Start soil changes in `sensor_demo/soil.py` and
`sensor_demo/surface.py`; see [SOIL_MODEL.md](sensor_demo/SOIL_MODEL.md).

The integration boundary is the rendered scene. Publish the evolving sand
surface into the stage before depth capture, and keep sand, camera and robot
poses on the same simulation step. Both calibrated cameras should observe that
surface through the existing depth/masking/FastDEM path. Keep direct soil-grid
ground truth separate from sensor-derived heightmaps. The existing ground-truth
soil publisher is described in [sensor_demo/README.md](sensor_demo/README.md).

The scanned sand remains visible geometry until explicitly replaced or masked;
adding a dynamic surface alone cannot reveal excavated holes hidden by the
original scan. Decide how to replace that region during integration. Also keep
the registered ground reference consistent (`--ground-z`, currently -0.0245 m
for the calibrated replay), instead of assuming the separate soil demo's origin.

The gamepad viewer uses the external learned arm and slew forward models as
free-space dynamics, integrating at 100 Hz on a CPU worker and applying snapshots
to USD before camera capture. It does not run PhysX. The released arm model's
carriage pitch is applied after bearing yaw, with the rocking asset's +/-3 degree
limit. All model channels and the 10 ms training timestep are checked at load.
Model source, weights and licenses stay in the actuator checkout.

Run `python tools/benchmark_gamepad.py` in the Isaac Lab environment to check
60/10/2 Hz input polling against the real models. Add `--render --headless --frames 80
--depth-out logs/gamepad/compact` to measure actual RTX load, `--show-viewport`
for a full viewport comparison, or `--frame-delay 0.5` to inject a slow render thread.
The report separates inference, scheduling lateness, dropped wall time, velocity
clipping and render-loop timing. It measures timing and numerical health, not
model accuracy against real-machine trajectories.

On 2026-11-27, the RTX 2000 Ada Laptop / i7-13700H run held 99.95 Hz
control while synthetic gamepad motion and compact IR/depth/maps produced 2.28
map updates/s. Enabling the full viewport gave 2.17 updates/s and 99.93 Hz
control. The idle GUI reached roughly 4 map updates/s with 99.97 Hz control.
All three runs reported zero dropped control time, zero velocity clipping and
no non-finite predictions. These short runs show timing feasibility, not a
guarantee of smooth display or real-time scheduling: the idle GUI had a 125 ms
p95 scheduling delay followed by fixed-step catch-up. Artifacts are under
`logs/gamepad/{compact,full-viewport,gui}/` on this machine. Input is still polled
by Kit's render loop; it is not an independent 20 Hz input stream.

The ROS viewer currently reads paused USD/kinematic replay poses and rejects a
playing physics timeline. Dynamics integration needs synchronized physics/USD
poses and capture timestamps before that restriction can be removed.

Preserve the existing output contract: 64x64 front and 160x160 surround maps,
rows decreasing X and columns decreasing Y, ground-relative int16 millimetres,
`-32768` unknown and uint16 ages with `65535` unknown. ROS exposes float metre
heights with NaN for unknown. The selected profile or dataset defines grid
bounds and resolution. Clear accumulated maps when resetting an episode or
changing the fixed-base reference.

For a sand integration smoke test, capture a baseline, dig and deposit in view
of both cameras, and compare raw depth, filtered maps and separate ground truth.
Check robot occlusion, unknown cells and map reset as well as the rendered image.

## Tests

```powershell
# Default heightmap, replay, asset, ROS-contract and calibration tests; no viewer.
uv run --no-sync python -m pytest -q

# Soil sensor geometry and expert motion, including the shared URDF consumers.
uv run --no-sync python -m pytest -q sensor_demo/test_lidar.py sensor_demo/test_expert.py

# Soil physics (CPU by default); surface/RGB tests can require CUDA.
uv run --no-sync python -m pytest -q sensor_demo
```

`pytest.ini` bounds discovery to the maintained mapping/calibration suite and
isolates it from the enclosing Isaac Lab conftest. Logs, archives and downloaded
third-party test suites are excluded. Run subsystem tests explicitly when
working on them. `newton/test_teleop.py` uses Linux input APIs and runs on Linux.
`ros2/test_arm_filter.py` is a standalone ROS integration script, not a default
pytest test; follow its module instructions on a configured ROS machine.

The FastDEM round-trip test skips when the compiled adapter is absent.
`tests/test_robot_mapping.py` additionally needs the local robot checkout,
recording dataset and saved episode-002 comparison. Skips do not validate those
integrations. The default suite still needs Isaac Lab's Python dependencies,
including USD, but does not start Kit or ROS.

Preserve upstream asset/code licenses. Scratchpad-authored files should not
carry NVIDIA or Isaac Lab copyright headers.
