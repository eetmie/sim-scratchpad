# Generated output

This directory contains local runtime output, not source code or maintained tests.
Everything except this README is ignored by Git.

| Directory | Contents |
| --- | --- |
| `depth_mapping/` | Current depth capture, maps, profile and timing reports |
| `depth_replay/` | Replay plans, paired captures and offline comparisons |
| `3dgs/` | Viewer snapshots |
| `filter_sweep/` | Mapping parameter experiments |
| `camera_calibration/` | Calibration captures and reports |
| `rosbags/` | Local ROS recordings |
| `excavator_share_*/` | Previously exported sharing packages |
| `archive/` | Old diagnostic scripts and loose outputs, retained for reference |
| `fastdem/` | Legacy compiler/source/build cache; still supported |

New FastDEM builds go to `.cache/fastdem/`. Keep an existing `logs/fastdem/`
build in place: CMake records absolute paths. The runtime checks the new cache
first and then this legacy location. Use `--fastdem PATH` for a specific build.

Use a named subdirectory for each experiment, for example
`--depth-out logs/depth_mapping/sand-baseline`. Put reusable tools in `tools/`
and automated mapping tests in `tests/`. Run the capture inspector with:

```powershell
uv run --no-sync python tools/inspect_heightmaps.py logs/depth_mapping --plot
```

Replay plans and saved captures are inputs to later runs as well as outputs.
Copy them before removing old runs. Archived scripts are historical snapshots;
they may contain old paths and are not supported entry points.

The initial cleanup preserved 56 loose files in
`archive/2026-11-27-loose-files/`. Its `manifest.json` records each original path,
archive path, size and SHA-256 hash; copy an archived file to its original path
if needed. Maintained replacements belong in source control, not this archive.
