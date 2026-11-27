"""Simulator-owned camera/map configuration, with explicit recorded-dataset overrides."""

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config/heightmaps"
STREAMS = {"cabin": "cam2", "arm": "cam1"}


def configuration_arguments(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="Camera/map/filter profile directory; default: config/heightmaps")
    source.add_argument("--dataset", type=Path, help="Use recorded dataset calibration and the robot checkout's policy")


def configuration_files(dataset=None, session=0, *, config=None):
    if dataset is not None and config is not None:
        raise ValueError("Choose either a simulator config or a recorded dataset")
    if dataset is not None:
        dataset = Path(dataset)
        return dataset / "meta/heightmaps.json", {
            key: dataset / f"native/session-{session:03d}/{key}.depth.json" for key in STREAMS.values()
        }
    if session != 0:
        raise ValueError("A session index requires a recorded dataset")
    directory = Path(config) if config is not None else DEFAULT_CONFIG
    return directory / "heightmaps.json", {key: directory / f"{name}.json" for name, key in STREAMS.items()}


def reduced_depth_sensor(sensor, divisor):
    """Preserve optical FOV, pixel centres, depth units and stored orientation."""
    intr = sensor["depth_intrinsics"]
    if divisor not in (1, 2, 4) or intr["width"] % divisor or intr["height"] % divisor:
        raise ValueError("Depth divisor must be 1, 2 or 4 and divide both sensor dimensions")
    if divisor == 1:
        return sensor
    scaled = dict(intr, width=intr["width"] // divisor, height=intr["height"] // divisor)
    for key in ("fx", "fy"):
        scaled[key] = intr[key] / divisor
    for key in ("ppx", "ppy"):
        scaled[key] = (intr[key] + 0.5) / divisor - 0.5
    return {**sensor, "depth_intrinsics": scaled}


def load_profile(dataset=None, session=0, *, config=None):
    metadata, files = configuration_files(dataset, session, config=config)
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    sensors = {}
    for name, key in STREAMS.items():
        sensor = json.loads(files[key].read_text(encoding="utf-8"))
        intr = sensor["depth_intrinsics"]
        if sensor["rotation_deg"] not in (0, 180) or any(intr["coeffs"]):
            raise ValueError("Expected rectified D435i depth with 0 or 180 degree image rotation")
        dimensions = [intr[k] for k in ("width", "height", "fx", "fy")] + [sensor["depth_scale_m"]]
        if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
            raise ValueError("Invalid D435i intrinsics or depth scale")
        sensors[name] = {"key": key, **sensor}
    sources = {"heightmaps": metadata, **{f"camera/{name}": files[key] for name, key in STREAMS.items()}}
    return {
        "sensors": sensors,
        **{key: meta[key] for key in ("front", "surround", "map_hz", "fps", "height_color_limit_m")},
        # Native datasets use their own ground origin; the simulator needs the stage registration.
        "ground_z_stage_m": meta.get("ground_z_stage_m", -0.0245),
        "configuration_source": str(metadata.resolve()),
        "configuration_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
    }
