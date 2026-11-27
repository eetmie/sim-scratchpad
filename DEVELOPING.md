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

If your robot checkout already has a FastDEM adapter built for this machine in
`native_stack/tools/heightmap_bench/`, the viewer can use it directly.
Otherwise build it once from the scratchpad.

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

## Headless performance journal, 2026-11-27

Isaac Sim 6.2 on an RTX 5070 Laptop GPU (75 W maximum), with local RViz subscribed. Each replay
run visited the same 280 poses with two 640×480 depth cameras, two 224×168 IR
cameras, robot filtering, FastDEM and ROS. The receiver matched 180 observations
by stamp across both images, both camera infos, TF, the host clock and both maps.
Rates below use receiver wall time. Latency is the median from acquisition start
to arrival of both IR images. GPU busy is the steady-run mean from NVIDIA SMI.

| Replay experiment | Pairs/s | Latency | GPU busy | Finding |
| --- | ---: | ---: | ---: | --- |
| Original serial loop | 11.7 | 82 ms | 45% | Per frame, Kit spent 47.8 ms rendering, followed by 6.2 ms readback, 12.2 ms projection and 10.8 ms fusion. |
| Overlap mapping and publishing with the next render | 16.7 | 86 ms | 59% | A single ordered worker kept FastDEM history and publication in order. The higher throughput cost 4 ms of median latency. |
| Also skip color shading on the depth products and reduce host buffer copies | 25.5 | 67 ms | 52% | GPU frame time fell from 36.1 to 19.8 ms. Kit render time fell to 29.8 ms and readback to 4.4 ms. Both IR products kept their lighting. |

Times in milliseconds below are median Kit render times.

| Idea tested | Outcome |
| --- | --- |
| Remove the depth products' color output variable | 46.6 → 47.1 ms. RTX still ran the color passes. |
| Batch the cameras into tiled products | 46.6 → 36.0 ms. The arm IR image shifted and depth differed at fixed surfaces, so this path was discarded. |
| Render depth and IR on alternating Kit ticks | 47.4 → 138.8 ms per complete set. Switching renderer settings twice per observation outweighed the GPU saving. |
| Replace the synchronized Replicator step with a Kit tick and relax the render wait | 27.6 ms per tick, but only 6 of 32 alternating poses matched the camera output. The synchronized step matched all 32 on all four streams. |
| Select RTX Minimal shading only on the two depth products | 46.7 → 28.8 ms in the stationary probe. The GPU trace showed both depth color passes disappear. Moving test plates at 105 and 146 mm verified optical Z within one Z16 count and the CAD mask on both cameras across 48 frames. |
| Limit Kit to 4, 8 or 16 CPU threads | 25.79, 25.79 and 25.85 completed maps/s respectively on the same 192-pose replay. Kept the default of 16. |
| Disable presentation or enable async low-latency rendering | With deliberately alternating poses, standard rendering produced 26.4 maps/s and disabling presentation gave 26.6. Both matched 96/96 poses on every camera stream. Async reported 27.9 maps/s but matched only 23/96. Kept synchronous capture. |

At the last replay exposure, both camera poses, both map transforms and the ROS
stamp matched exactly. Median absolute height differences on cells observed in
both runs were 1 mm for the front map and 2 mm for the surround map. The Gaussian
splat depth renderer also varies across repeated captures at a stationary pose.
The depth products use RTX Minimal's no-color mode for non-color render outputs.
[NVIDIA RTX Minimal renderer reference](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_minimal.html)

For a comparable run after adding soil, use the normal launcher in the simulator
environment with RViz connected. `--uncapped` removes the wall-time capture limit
while preserving replay timestamps and the configured data clock.

```bash
bash ros2/run_latest.sh --headless --mode replay --ros-cameras --ir-width 224 \
  --uncapped --frames 280 --depth-out logs/ros-rviz/pre-soil
```

Use a new output directory for the soil run. For live viewing add `--ros-time wall`
and omit `--frames`. Two 25-second checks received 673 and 675 observations.
Recorded motion advanced at 1.0001× and 1.0002× wall speed across a replay loop.

Use `performance.json` for median and p95 stage times. `period_ms` measures the
interval between completed maps. `capture_to_publish_ms` includes mapping and ROS
publication for the same exposure. Overlapped stages cannot be summed to obtain
the frame interval. Raw traces and receiver measurements for this session are in
`logs/ros-rviz/headless-timing/`.

### Gamepad and live RViz sensor rates

A 12-second receiver trace of the previous live replay measured 26.8 Hz on both
IR cameras and both heightmaps. The source pose changed just 5.0 times/s, and
RViz had a 20 Hz display limit. The gamepad panel previously captured both IR
previews only when a 5 Hz map update was due.

Disabling and re-enabling the two Hydra depth textures at map cadence cost
about 100 ms per map render. Both a GUI gamepad test and a headless RViz receiver
then saw only 9 Hz IR, despite 5 Hz maps and 100 Hz hydraulic control.

Keeping the depth products allocated raised the 224×168 Kit IR panel to 27.0 Hz
with 640×480 depth, 5.1 Hz maps and 100.0 Hz control. Disabling Kit present
pacing gave no gain. Moving the depth annotator buffers to CUDA gave less than
1 Hz. DLSS Performance reduced IR render time but changed image intensity by
about 3–4 gray levels on average in held poses. A one-bounce limit darkened the
scene substantially. A native multi-tick experiment on the paused replay timed
out waiting for the first complete camera buffers. That scheduler normally
uses a renderer clock advanced by physics. The depth-resolution trials kept
the calibrated IR lighting and the synchronous capture path.
[NVIDIA multi-tick rendering reference](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_multitick_rendering.html)

Both cameras saw the same 110 recorded poses in each uncapped depth-size run.
IR stayed at 224×168. The robot projector sampled a 160×120 grid from each
depth camera using scaled focal lengths and principal points. Height errors
are absolute differences on cells known in both runs, pooled across all poses.

| Depth per camera | Projection stride | Median render | Complete pairs/s | Front error, median / p95 | Surround error, median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 640×480 | 4 | 29.5 ms | 25.9 | reference | reference |
| 320×240 | 2 | 23.9 ms | 29.6 | 1 / 6 mm | 3 / 14 mm |
| 160×120 | 1 | 22.6 ms | 30.9 | 2 / 7 mm | 3 / 14 mm |

The second native run reproduced every height cell and averaged 25.6 pairs/s.
Front coverage ranged from 48.58% to 48.68% and surround coverage from 17.16%
to 17.21%. At 320×240, the overlap of known cells divided by their union was
94.0% for the front and 98.0% for the surround. Coverage totals alone hide
changes at edges. The live demo uses 320×240. Select `--depth-divisor 4`
to trade another millimetre of median front error for 1.3 ms of render time.
Recorded comparisons and raw depth streams default to their native calibration.

To repeat the accuracy and throughput comparison before or after adding soil,
run the example with divisors 1, 2 and 4 and use a different output path for each.

```bash
bash ros2/run_latest.sh --headless --mode replay --ros-cameras --ir-width 224 \
  --uncapped --depth-divisor 2 --frames 110 --depth-out logs/ros-rviz/soil-baseline-320
```

Batching the USD link transform edits into one change notification saved about
0.8 ms per moving arm pose. Polling the ROS executor cost another 3–4 ms per
camera tick in Kit. The live publisher now polls it on the mapping worker at
map cadence. DDS still publishes an exposure and TF at every camera tick.

| Final wall-time check | Both IR cameras | Both maps | Motion |
| --- | ---: | ---: | --- |
| Kit gamepad panel, 480 frames | 29.88 Hz | 5.00 Hz | Learned hydraulic models at 99.97 Hz with scripted valves, zero dropped control time. IR interval p95 was 33.7 ms. |
| Kit gamepad panel and RViz together, including both raw depth topics | 28.71 Hz | 5.01 Hz | Raw 320×240 depths and camera infos arrived at the map stamps. |
| Headless replay with local RViz, 25 seconds | 29.88 Hz | 5.00 Hz | Recorded motion advanced at 1.0003× wall speed. Joint values changed 28.84 times/s. IR arrival interval p95 was 40.7 ms. |

The demo checks used 224×168 IR and 320×240 depth. The RViz receiver
matched all 748 IR pairs to camera info and TF, and all 125 map pairs to
the corresponding IR exposure. Mean GPU busy during a 12-second sample
was 46%. The local 660-exposure motion file comes from the same episode
as the bundled 110-pose example, generated with `--stride 1 --timestamps ir`.
Both captures and poses are measured in wall time.

At 640×480 IR with 160×120 depth, the simulator captured 25.25 Hz. The local
best-effort receiver got 16.1–16.3 Hz per image topic with RViz subscribed,
while both maps still arrived at 5.00 Hz. Full-size IR has a separate render
and delivery cost. The measurements and map comparison are under
`logs/ros-rviz/robot-rates/`.

### RViz terrain cloud at the depth rate

With RViz subscribed, two successive loops of the same 22-second motion used
the old 2 Hz cap and the new default of one cloud per completed map. Each loop
contributed 106 map updates after excluding the first and last two. The receiver
measured wall time. Payload rates exclude DDS and network framing.

| Cloud setting | Clouds received | Surround map | Boom IR | Cloud payload | Map publication, median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 Hz cap | 35 at 1.67 Hz | 5.00 Hz | 29.86 Hz | 0.087 MB/s | 0.99 / 2.44 ms |
| Every depth/map update | 106 at 5.01 Hz | 5.01 Hz | 29.53 Hz | 0.264 MB/s | 1.50 / 4.00 ms |

The old half-second interval first expired on the third 200 ms map tick. In the
full-rate run, all 106 clouds matched their surround map's acquisition stamp.
Within the capped run, ticks with a cloud took 1.45 ms median to publish all map
messages, versus 0.95 ms on ticks without one. Both paths use the existing fused
grid. The cloud averaged 4,395 known cells at 12 bytes each in the full-rate run.
Higher terrain coverage increases traffic in proportion to the known-cell count.

Turning off `publish_map_cloud` through ROS delivered zero clouds while 16 maps
and 91 boom IR frames arrived. Re-enabling it delivered 16 clouds with 16 matching
map stamps. Unchecking the terrain display in the running RViz dropped the cloud
subscriber count to zero while the map kept updating. Checking it restored the subscription.
In a final 25-second check after restoring the display, the receiver saw 749 pairs
of IR frames (29.95 Hz per camera) and 124 sets of front map, surround map and cloud
(5.00 Hz each). Every set matched an IR exposure and TF by stamp. Recorded motion
advanced at 0.999× wall speed. Raw depth stayed local throughout.
[RViz and ROS controls](ros2/LATEST.md#topics-and-numeric-conventions)

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
