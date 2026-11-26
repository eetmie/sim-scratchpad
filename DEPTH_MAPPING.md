# D435i depth and terrain mapping

The 3DGS viewer can render depth from the cabin and arm D435i mounts and build
terrain maps with the layout in `heightmap_tests_complete`. Run from this checkout
using Isaac Lab's existing environment:

```powershell
uv run --no-sync python launch_3dgs.py --depth-maps --dataset C:/Users/sh23937/Desktop/heightmap_tests_complete
```

The **D435i depth and terrain maps** window shows both depth images, the front
map and the accumulated surrounding map. **The default is now the real robot's
masking, cabin-first fusion and FastDEM**, shared with the offline comparison
through `robot_mapping.py`. Capture uses full 640x480 images; the robot's stride-4
projection produces 160x120 depth/mask previews. Red marks self-filtered pixels.
The maps target the robot profile's 5 Hz, subject to GPU and capture time.
The sensors follow the live USD link transforms. Physics
starts paused; rendering depth preserves the current playback state. Moving the
fixed base clears the surrounding map to avoid retaining cells in the wrong frame.

Use **Save current depth + maps** to export the current result and **Clear
accumulated map** to restart coverage. Closing the viewer also saves the final
capture. All outputs default to `logs/depth_mapping/`; use `--depth-out` to change
that location. Camera and render-product changes are authored in the session
layer. The dataset is read-only.

Loop the saved recorded motion immediately, without rerendering the offline
comparison first:

```powershell
uv run --no-sync python launch_3dgs.py --depth-maps --dataset C:/Users/sh23937/Desktop/heightmap_tests_complete --replay logs/depth_replay/episode-002/replay.json
```

The map clears when the replay wraps. Omit `--replay` for the ordinary scene.
Use `--robot-repo PATH` or `KAIVURI_REPO` for another checkout and `--fastdem PATH`
for another compiled adapter. Defaults find this machine's robot checkout and
portable build. `--mapping preview` explicitly selects the old CAD-ID-filtered,
lightweight mapper. The RTX LiDAR Script Editor helper remains a separate sensor
prototype; the D435i and FastDEM demo uses stereo-camera depth.

For one headless capture:

```powershell
uv run --no-sync python depth_mapping.py --dataset C:/Users/sh23937/Desktop/heightmap_tests_complete --out logs/depth_mapping/capture
```

## Sensor calibration and depth

The tools read `native/session-000/cam2.depth.json` for the cabin and
`cam1.depth.json` for the arm. Each uses its own recorded focal lengths, principal
point, image size, depth scale and image rotation. The existing fitted IR cameras
are preserved. New sibling depth cameras compose those fitted IR mounts with the
SDK's depth-to-IR extrinsics. This includes the arm's 49.896 mm stereo baseline.
The arm's recorded image orientation is rotated 180 degrees; saved transforms
still describe the physical, unrotated optical axes: right, down, forward.

RTX's `distance_to_image_plane` supplies optical Z, not ray length. Captures are
quantized to the dataset's approximately 0.1 mm Z16 scale. Zero means no return
or depth beyond the Z16 encoding range. Raw depth retains the CAD robot. For
mapping, the robot's mesh-distance margins, projected arm silhouette, camera-body
exclusion and arm-view/forward rules remove self-hits after rendering, retaining
their occlusion. CAD instance IDs are only requested for diagnostic raw-depth
comparisons or the legacy preview; they are not inputs to the robot mapper.
Both cameras contribute with the robot's range, annulus and height crop.

## Map format and exports

Map geometry comes from `meta/heightmaps.json`:

| Output | Layout | Frame |
| --- | --- | --- |
| `front_mm` | 64x64, X 0.1..0.9 m, Y -0.4..0.4 m, 12.5 mm cells | Level frame at the slew pivot, current cabin yaw |
| `surround_mm` | 160x160, XY -0.8..0.8 m, 10 mm cells, radius 0.14..0.8 m | Level undercarriage frame |
| `surround_age_updates` | uint16, accepted paired map updates since observation | Same surrounding grid |

Rows decrease X and columns decrease Y. Heights are int16 millimetres, with
`-32768` for unknown; unknown age is `65535`. Grayscale previews match the dataset:
ground 128, -120 mm 16, +120 mm 240, unknown black. No holes are inpainted.

`latest.npz` contains one capture: the three grids above, `depth_cabin_z16` and
`depth_arm_z16` (uint16, 480x640), `optical_to_world_cabin` and
`optical_to_world_arm` (column-vector 4x4 matrices, metres), filtered
`points_world_m` (Nx3, sampled at the profile's `mapping_pixel_stride`), and
`stamp_s` (elapsed wall time). Unlike the dataset's
chunks, these arrays have no leading time dimension. PNG files contain both
lossless Z16 images and grayscale map previews. `profile.json` records calibration,
units, policy, source hashes, map geometry and the depth model. Live outputs also
contain strided `self_mask_cabin`/`self_mask_arm` (0 missing, 1 return, 2 rejected).
`surround_age_frames` remains a compatibility alias for update ages in robot mode;
the preview still counts nominal 30 Hz dataset frames. Read the profile's units.
`diagnostics.json`, `updates.jsonl` and `performance.json` separate rendering,
readback, filtering and FastDEM time. The world-point export contains filtered
camera candidates before arm-view gating and cabin-priority fusion.

The measured headed replay on this laptop (35 warmed samples, full 640x480
capture) completed about 3.83 map updates/s against the 5 Hz target. Median
rendering was 76 ms, readback 4.5 ms, projection/masking 10.6 ms and FastDEM
13.8 ms. Depth-preview preparation/upload added about 149 ms; point export was
below 1 ms. The GUI remains the immediate performance target. The standalone
real-data sweep's current policy took about 66 ms/update on this laptop, without
Isaac rendering or UI. These are laptop measurements, not a Jetson throughput
claim. `OPENBLAS` is limited to one thread during mapping, using a cached
controller; this matches the robot launcher and avoids repeated DLL discovery.
The live loop also avoids a duplicate viewport render when Replicator captures.

## Fidelity limits

This is ideal rendered depth from the static Gaussian room scan and live CAD
geometry. It does not reproduce passive-stereo matching failures, RealSense
disparity shift/noise, IR material response or the recording's empirical depth
correction. The mapper itself uses the robot's FastDEM Kalman estimator,
ray clearing and arm-view rejection. Simulated depth remains ideal; recorded real
depth uses its empirical correction in offline experiments. Live USD pivots and
tool geometry feed the same FK-based masks. Slew uses the controller's quaternion
swing/twist convention rather than Euler heading, which changes with tilt.

Heights use a fixed sand reference in stage coordinates. The default
`--ground-z -0.0245` follows the existing native-to-stage replay offset; the
Gaussian reconstruction's surface can differ from that reference by centimetres.
Use `--ground-z` to select a measured reference for another scene. The captured
room scan has fixed sand: these tools observe changes to rendered geometry, but
do not add soil dynamics or replay recorded depth as simulated measurements.

Focused CPU validation:

```powershell
uv run --no-sync python -m pytest -q --noconftest test_depth_mapping.py
```

## Replay logged motion and compare recorded depth

Use the same logged FK/IMU replay that calibrated the IR cameras, now sampled
at each camera's depth host timestamp. This drives the USD links kinematically
with physics paused. It preserves the existing CAD frame corrections and room
registration; it does not replay valve commands through a dynamics controller.

```powershell
uv run --no-sync --with av python replay_depth.py --dataset C:/Users/sh23937/Desktop/heightmap_tests_complete --episode 2 --start 12 --stop 672 --stride 6 --out logs/depth_replay/episode-002 --loop
```

This compares 110 paired samples over a 22-second held-out recording segment.
`--with av` supplies the H.264 video encoder through a temporary uv overlay.
The cabin and arm are each rendered at their own depth exposure's interpolated
link pose. Their native Z16 frames are read in recorded pixel orientation, and
the appropriate session supplies the sensor calibration. Joint angles are also
interpolated for the motion plot. Exposures outside the available valid FK time
range are rejected instead of silently accepting a clamped pose.

The comparison writes:

- `comparison.mp4`: real depth, simulated depth and signed error for both cameras,
  played at the recorded time scale. Depth colors cover 0.1--0.9 m; error colors
  cover -100..+100 mm; black marks missing or out-of-range pixels.
- `comparison-*.png`: first, middle and final paired comparisons.
- `errors_and_angles.png`: per-frame median/p90 error and the logged joint motion.
- `summary.json` and `frames.csv`: error, bias and coverage statistics. The summary
  includes all common valid pixels and a separate result outside the simulated
  CAD robot mask. That mask does not identify real hardware absent from the CAD.
- `cabin/frame-*.npz` and `arm/frame-*.npz`: rendered `z16`, `robot_mask`, optical
  `pose` and matched `timestamp_ns`. Recorded frames stay in the source dataset.
- `replay.json`: exact applied transforms, exposure timestamps and logged angles.

Errors are **sim minus real optical Z**, in millimetres, on common valid pixels
inside the 0.1--0.9 m radial working volume. Coverage is reported separately so
missing real returns are visible. Pooled median/p90 values use 1 mm histogram
bins. The comparison applies no fitted image alignment, depth offset, scale or
empirical real-depth correction. Sand deformation and geometry or material
differences between the recording and the static scan remain part of the result.

`--loop` leaves the viewer replaying the selected motion with live depth/maps
after the comparison finishes. Omit it to close after exporting, or replace it
with `--headless` for batch use. Lower `--stride` for denser sampling.

## Apply the real robot's mapping masks offline

Use the current `kaivuriprokkis` checkout as the single owner of projection,
robot exclusion and camera fusion. This tool reads the saved comparison frames;
Isaac Sim does not need to render them again:

```powershell
uv run --no-sync python offline_robot_maps.py --robot-repo C:/Users/sh23937/Documents/GitHub/kaivuriprokkis --dataset C:/Users/sh23937/Desktop/heightmap_tests_complete --replay logs/depth_replay/episode-002 --out logs/depth_replay/episode-002/robot_masks
```

The bridge imports `native_stack/tools/heightmap_bench/mapper.py` directly and
loads its `config/heightmap_pipeline.json` and arm distance-grid model. It uses
the same `DepthProjector` and `update` functions for real and simulated data:
mesh-distance and projected-silhouette rejection, other-camera exclusion,
arm-view/forward gates and cabin-priority filling. Masks are saved separately
from raw Z16. Real depth uses its recorded empirical correction; ideal simulated
depth uses the actual saved USD camera pose with that correction disabled.

`settings.json` records the exact robot revision and current policy. The original
dataset does not record its full projection policy, so this is a reproducible
comparison under the current policy, rather than a claim of exact reproduction
of the dataset's already-generated maps. `masks-middle.png` visualizes rejected
pixels, `fronts-middle.png` compares masked front maps, and `frame-*.npz` saves
their numeric grids and masks. `summary.json` also compares raw optical depth
outside the **union** of the two self masks, using every fourth stored pixel.
The initial replay's unmasked comparison remains available separately.

`real.scans` and `sim.scans` are ready-to-run binary inputs for the robot's
ROS-free `fastdem_replay` adapter. They preserve optical-frame points and each
sensor's 3x4 pose, including the origin required for clearing rays. On the
machine with the compiled adapter:

```bash
fastdem_replay fastdem.yaml < real.scans > real.maps
fastdem_replay fastdem.yaml < sim.scans > sim.maps
```

The output for each update is 160x160 little-endian float32 heights followed by
160x160 uint16 ages (153,600 bytes total). Input layout matches the robot adapter:
uint32 scan count, then per scan uint32 point count, 12 float64 row-major pose
values and Nx3 float32 optical points. A native Windows build of that adapter
must put stdin and stdout into binary mode.

Pass `--fastdem PATH` to a compatible executable to generate surround maps
during this offline run. Without it, the tool exports masked front maps and
FastDEM inputs; it does not substitute another estimator for FastDEM's surround
map. The simulator's live viewer now selects the shared robot mapper by default.

### Windows build and filtered comparison

Build the same pinned core with a portable LLVM-MinGW compiler; no administrator
installation or WSL is required:

```powershell
uv run --no-sync --with cmake --with ninja python build_fastdem.py --llvm-mingw --robot-repo C:/Users/sh23937/Documents/GitHub/kaivuriprokkis
uv run --no-sync --with av python offline_robot_maps.py --robot-repo C:/Users/sh23937/Documents/GitHub/kaivuriprokkis --dataset C:/Users/sh23937/Desktop/heightmap_tests_complete --replay logs/depth_replay/episode-002 --out logs/depth_replay/episode-002/robot-main-fastdem --fastdem logs/fastdem/build-mingw/fastdem_replay.exe --video
```

`build_fastdem.py` keeps the compiler, third-party source and licenses under
`logs/fastdem`. It verifies the compiler archive against the published SHA-256,
pins FastDEM at `5cd3e25` and nanoGrid at `058a6a6`, applies the robot's core patch,
and builds the robot's C++ adapter with Windows binary stdin/stdout. The build
manifest records source hashes and tool versions. The compiler also remains
available at `logs/fastdem/llvm-mingw-20261117-ucrt-x86_64/bin/`; this does not
replace a future MSVC/Windows SDK installation for CUDA or MSVC builds.

The filtered comparison includes `surround_real_mm`, `surround_sim_mm` and their
uint16 `surround_*_age` grids in every per-frame NPZ. Ages count accepted paired
map updates, with 65535 for unknown. Both estimators start empty at the selected
segment's beginning, so earlier recording history is absent. Each archive also
stores the two exposure timestamps in cabin/arm order.

`filtered-comparison.mp4` shows the robot's masks on both real/sim depth streams,
current fused front maps, accumulated FastDEM surrounds and signed map errors.
Playback follows the median selected exposure cadence. Heights use the dataset's
grayscale (+/-120 mm); error colors cover +/-50 mm; black means unknown. First,
middle and final snapshots are saved as `filtered-*.png`, and
`surrounds-middle.png` provides a labeled quantitative view. `summary.json`
reports front and surrounding errors only where both maps have observations.
The separate optical-depth errors remain raw Z, without the real-depth correction
that is applied when making the real map.

Focused checks, including a round trip through the compiled C++ adapter:

```powershell
uv run --no-sync python -m pytest -q --noconftest test_fastdem.py test_replay_depth.py test_depth_mapping.py camera_calibration/test_calibration.py
uv run --no-sync python ros2/arm_grid.py
```

Verified comparison on episode 002, frames 12..666 at stride 6 (110 paired
updates, approximately 22 seconds), using robot revision `1412ea0`:

| Compared map | Median absolute error | 90th percentile | Real observed cells also observed in sim |
| --- | --- | --- | --- |
| Front | 5.5 mm | 11.5 mm | 95.4% |
| FastDEM surround | 5.5 mm | 13.5 mm | 98.9% |

These pool commonly observed cells at every update; a remembered surround cell
can contribute more than once. Quantiles use 1 mm bins. The static room scan,
ideal rendered depth and existing camera calibration are retained without a
fitted depth offset or scale. Eight focused tests pass, including the compiled
adapter's flat-ground response, persistence and update ages; the pulled
`arm_grid.py` geometry check also passes.

## Offline filter experiments on the real recording

```powershell
uv run --no-sync python benchmark_robot_maps.py --dataset C:/Users/sh23937/Desktop/heightmap_tests_complete --episode 0 --stop 1812 --sim-replay logs/depth_replay/episode-002 --out logs/filter_sweep/heldout-flat-000-optimized
```

This uses 300 paired updates over 60 seconds of the held-out flat recording.
It evaluates a fixed undercarriage ROI, X 0.20..0.65 m, Y +/-0.25 m. Temporal
standard deviation uses identical observed cells at identical times for all
profiles, with at least five observations per cell. It does not confuse cabin
rotation with changing terrain or count retained stale cells as fresh measurements.
This measures repeatability on that take, including calibration/FK and stereo
variation, rather than an independent ground-truth accuracy measurement.

| Private test profile | Boom / arm / bucket margins | Silhouette growth | Mean current ROI coverage | Median temporal spread |
| --- | --- | --- | --- | --- |
| Current wide | 90 / 45 / 100 mm | 2 strided cells | 64.07% | 2.355 mm |
| Extra wide | 120 / 60 / 120 mm | 3 strided cells | 63.94% | 2.354 mm |
| Smaller distance margins | 60 / 35 / 80 mm | 2 strided cells | 64.28% | 2.362 mm |
| Wide, less silhouette | 90 / 45 / 100 mm | 1 strided cell | 64.12% | 2.355 mm |

All four profiles retained zero labeled CAD robot points after fusion in the
separate 110-sample simulated clip. That diagnostic checks geometric self-hits;
ideal rendered depth does not reproduce real stereo flying pixels, cables or
hardware missing from CAD. The real-data sweep supports retaining the existing
wide baseline: extra exclusion did not measurably improve repeatability here,
and narrower margins recovered very little coverage. Robot defaults are unchanged.

`report.json` records private overrides, source hashes, coverage, fixed-support
plane residuals, temporal spread and compute time. `filter-tradeoffs.png`
summarizes the comparison. Compute times exclude depth rendering, video and
metric-grid reconstruction. Isaac logs those costs separately. The next useful
noise experiment is temporal/outlier handling on both flat and digging takes,
checking response to a newly dug hole as well as stability on unchanged sand.

## Running the physical robot

The desktop export's `native/session-000/session.json` records native dual IR
and **both raw depth streams**, with real PWM and IMUs, slew enabled and tracks
disabled. It predates heightmap-only recording. The recorded sensor metadata
shows 640x480, 30 FPS, 100 micrometre depth units and arm disparity shift 25.
An equivalent capture command on the robot is:

```bash
cd ~/kp-depth-close/native_stack
python3 tools/record_native.py --session "$HOME/data_collection/demo-$(date +%s)" --task "dig the sand" --enable-slew --depth both --depth-units-um 100 --cam1-disparity-shift 25
```

The exact historical shell command is not saved in the export; gamepad paths and
session directory names can differ. For the current workflow that records the
robot's filtered/fused maps directly, use:

```bash
python3 tools/record_native.py --session "$HOME/data_collection/hm-demo-$(date +%s)" --task "dig the sand" --enable-slew --depth heightmaps
```

That mode gets capture and filtering settings from `config/heightmap_pipeline.json`
and keeps FastDEM alive across takes in one recorder session. For a stationary
mapping-only performance demo with the real cameras and IMUs, the existing
`tools/heightmap_bench/bench_live.py --mode fused` runs its own dry controller:

```bash
export PYTHONPATH=tools:tools/heightmap_bench OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
h=tools/heightmap_bench
python3 "$h/bench_live.py" --native . --out "$HOME/heightmap-demo-$(date +%s)" --mode fused --seconds 60 --calibration "$h/calibration.json" --fastdem "$h/fastdem_replay" --config "$h/fastdem.yaml"
```
