# Developing the simulator

This is an independent repository inside Isaac Lab. Use the enclosing Isaac Lab
Python environment (`uv run --no-sync`) and run commands from this checkout.
Fetch LFS assets before starting the viewer. The main heightmap entry point is
`launch_3dgs.py --depth-maps`; the Newton soil demo has its own entry point.

## Find the code

| Area | Source and responsibility |
| --- | --- |
| Viewer | `launch_3dgs.py`: app lifecycle, replay, capture scheduling, optional ROS |
| Simulated depth | `depth_mapping.py`: calibrated cameras, render/readback, maps and exports |
| Robot mapping | `robot_mapping.py`: bridge to the external robot policy and FastDEM |
| Calibration | `camera_calibration/`: camera mounts, recorded poses and calibration tools |
| Offline comparisons | `replay_depth.py`, `offline_robot_maps.py`, `benchmark_robot_maps.py` |
| Simulated soil | `sensor_demo/soil.py`: soil state; `surface.py`: rendered surface; `demo.py`: simulation loop |
| Shared viewer helpers | `newton/`: input, cameras and mesh publishing; `sand.py` is the older lightweight sand model |
| ROS transport | `ros2/viewer_bridge.py`, `viewer_scene.py`, `map_contract.py`; see [LATEST.md](ros2/LATEST.md) |
| Shared assets | `assets/excavator_bucket.usd`, `assets/urdf/`, `assets/simready/` |
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

The `kaivuriprokkis` checkout owns the mapping algorithms and default parameters.
Set `KAIVURI_REPO` or pass `--robot-repo`; do not copy its filter implementations
into the simulator. Pass the recording directory with `--dataset`. The dataset
supplies camera calibration and map geometry and is read-only.

Build FastDEM once on each development machine:

```powershell
uv run --no-sync --with cmake --with ninja python build_fastdem.py --llvm-mingw --robot-repo C:/path/to/kaivuriprokkis
```

On Linux omit `--llvm-mingw`. New builds use `.cache/fastdem/`; the runtime falls
back to the existing `logs/fastdem/` build on older checkouts. Do not move a
configured CMake tree: its cache contains absolute paths. `--work` changes the
build location and `--fastdem` selects a particular executable.

```powershell
uv run --no-sync python launch_3dgs.py --depth-maps --dataset C:/path/to/heightmap_tests_complete --robot-repo C:/path/to/kaivuriprokkis --depth-out logs/depth_mapping/sand-baseline
uv run --no-sync python tools/inspect_heightmaps.py logs/depth_mapping/sand-baseline --plot
```

Use the viewer's save button or close it to export a capture. Detailed formats,
replay commands and experiments are in [DEPTH_MAPPING.md](DEPTH_MAPPING.md).

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

The ROS viewer currently reads paused USD/kinematic replay poses and rejects a
playing physics timeline. Dynamics integration needs synchronized physics/USD
poses and capture timestamps before that restriction can be removed.

Preserve the existing output contract: 64x64 front and 160x160 surround maps,
rows decreasing X and columns decreasing Y, ground-relative int16 millimetres,
`-32768` unknown and uint16 ages with `65535` unknown. ROS exposes float metre
heights with NaN for unknown. Dataset metadata remains authoritative for grid
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
