# Simulator camera and heightmap profile

The live viewer and single-capture command use this directory by default. No
recorded dataset is required. Select a different profile directory with
`--config PATH`; keep the same filenames and resolve `arm_model` relative to it.

| File | Owns |
| --- | --- |
| `cabin.json`, `arm.json` | D435i intrinsics, depth-to-IR extrinsics, image orientation and Z16 scale |
| `heightmaps.json` | Map geometry, calibration, stage ground reference, tool tip and arm-mask asset path |
| `pipeline.json` | Projection stride, robot/camera masks, arm-view gates and runtime rates |
| `fastdem.yaml` | FastDEM fusion, ray clearing and terrain-grid settings |

These are the validated settings from the session-000 camera calibration and
the robot revision identified in `heightmaps.json`. Camera serials identify the
calibration pair and satisfy the shared projector's consistency check; they do
not connect to real devices. Hardware disparity settings are calibration
metadata, not a simulated stereo-noise model. Empirical real-depth correction
is deliberately empty for rendered depth. Fitted IR mounts remain in
`demo_env.usd`; the viewer composes them with the depth-to-IR extrinsics here.

The arm distance grid is `assets/arm_model.npz`, shared with the ROS self-filter.
The robot checkout still supplies the mapping implementation and FastDEM build
sources. Changes to that implementation can change behavior even with fixed
settings; capture profiles record its revision and source hashes along with
the selected configuration hashes.

`--dataset PATH` explicitly selects recorded calibration and the robot checkout's
current policy for comparison work. It is mutually exclusive with `--config`.
Replay generation and real-versus-sim comparisons continue to require recordings.
Normal demos use `uv run --no-sync python launch_3dgs.py --demo`.

Edit `pipeline.json` for simulator filter experiments. The older ROS-side C++
mapping prototype in `ros2/mapping.launch.py` still has its own projection
parameters; the calibrated viewer's `--ros` mode publishes the simulator's maps.
