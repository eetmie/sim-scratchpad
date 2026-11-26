# Excavator IR camera calibration

## Calibrated stage for `heightmap_tests_complete`

Open `demo_env.usd` in Isaac Sim for the fitted cabin and arm mounts, factory IR intrinsics,
and registered Gaussian splat. The scene uses the original overhead light and grayscale response from the
first comparison video. Capture and replay use the recorder's image orientation, including the arm stream's
180° rotation.

```bash
uv run --no-sync python -m camera_calibration.capture \
  --out logs/camera_calibration/real_ir/still
uv run --no-sync python -m camera_calibration.recorded_poses ~/Desktop/heightmap_tests_complete \
  --episode 2 --start 12 --stop 672 \
  --out logs/camera_calibration/real_ir/replay.json
uv run --no-sync python -m camera_calibration.replay_ir \
  --plan logs/camera_calibration/real_ir/replay.json \
  --out logs/camera_calibration/real_ir/replay
```

Two 33 mm AprilTag interiors in episodes 000 and 001 constrained the camera mounts. Episode 002 was held out.
The fitted mounts project the held-out tag corners with median/p90 errors of 0.80/1.57 pixels for the cabin
and 2.92/6.00 pixels for the arm. In the two slew clips, the rendered moving tag corners differ by a median
of 0.93 and 1.46 pixels in the cabin. The arm's median is 3.22 pixels on the second turn, where tags are visible.
The replay interpolates measured FK at each IR timestamp. It uses the static room scan and CAD robot, so
the near-camera hardware, sand deformation, and IR reflectance can differ from the recording.

For timestamp-matched D435i depth comparisons and a looping robot replay, see
[Replay logged motion and compare recorded depth](../DEPTH_MAPPING.md#replay-logged-motion-and-compare-recorded-depth).
`recorded_poses --timestamps depth` uses the depth exposure clock with the same calibrated link-pose replay.

## Generic calibration workflow

Run from `scripts/sim-scratchpad`. These commands use Isaac Lab's existing `.venv`. Stage inspection, fitting and
layer authoring use CPU tools. Image capture uses Isaac Sim and an RTX GPU.

```bash
mkdir -p logs/camera_calibration
uv run --no-sync python -m camera_calibration.usd_cameras inspect > logs/camera_calibration/baseline.json
uv run --no-sync python -m camera_calibration.capture --out logs/camera_calibration/baseline
uv run --no-sync python -m pytest -q camera_calibration/test_calibration.py
```

The inspection lists the composed assets, Gaussian splat prim, and both camera poses. Capture writes one grayscale
PNG and a JSON sidecar per camera at the stage's authored pose, with physics paused. Use `--camera cabin` or
`--camera arm` for one view. The two tray tags are visible in the baseline cabin image. The bucket partially
occludes one tag in the arm view.

## Calibrate against another recording

Use original IR frames and the stream metadata for their exact capture mode. The native recorder stores the cabin
as `cam2` and the arm as `cam1`. Check the serial and selected IR index. The existing robot profile uses cabin IR1
and arm IR2, so a depth-frame mount may need the SDK's depth-to-IR extrinsics before use as an IR mount. The D435i
normally delivers hardware-rectified IR with zero distortion. Use the recorded profile values for each stream.
[RealSense projection and rectification guide](https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/)

```bash
uv run --no-sync python -m camera_calibration.usd_cameras inspect --native-session /path/to/session \
  > logs/camera_calibration/seed.json
```

The importer reads `cam1.depth.json` and `cam2.depth.json` and takes the IR intrinsics and recorded image rotation.
The stage mounts serve as initial estimates. In the JSON, `mount` means parent-link-relative position in meters
and an optical-to-parent quaternion in `[x, y, z, w]` order. The optical axes are right, down, forward. The
intrinsics are in pixels for the unrotated sensor image, including the principal point and Brown–Conrady
coefficients `[k1, k2, p1, p2, k3]` when present. Capture and reprojection apply `image_rotation_deg` to match the
recorded pixel orientation. The Isaac Sim viewport shows the unrotated sensor orientation.

If you have measured 3D target points, supply a CSV headed `x_m,y_m,z_m,u_px,v_px` with at least six well-spread
matches. Express points in the camera's parent link (`upper_carriage` or `tilt_boom`) at the image timestamp.
For a fixed world target observed while the joints move, use timestamp-matched FK to transform each point into the
link before fitting. Supply pixel coordinates from the recorded IR image. A second take or pose can serve as the
holdout. The fitter reports RMS and 90th-percentile reprojection error in pixels before and after the fit.

```bash
uv run --no-sync python -m camera_calibration.fit_mount logs/camera_calibration/seed.json arm \
  /path/to/arm-fit.csv --holdout /path/to/arm-holdout.csv --out logs/camera_calibration/fitted.json
uv run --no-sync python -m camera_calibration.usd_cameras apply \
  logs/camera_calibration/fitted.json logs/camera_calibration/calibrated.usda
uv run --no-sync python -m camera_calibration.capture \
  --stage logs/camera_calibration/calibrated.usda --out logs/camera_calibration/calibrated
```

Fit the cabin in the same way, or edit the JSON directly if measured mounts are available. Open the resulting
`.usda` in Isaac Sim to view the scene with the overrides. The overlay sublayers the original stage and authors
camera poses, apertures and RTX OpenCV pinhole parameters. Keep its relative path to the base stage intact.
[Isaac Sim OpenCV lens support](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_camera.html)

The grayscale capture approximates the appearance of the IR feed using rendered visible light. IR reflectance,
emitter speckle, exposure response and sensor noise still depend on the real captures.
