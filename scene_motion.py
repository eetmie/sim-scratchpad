"""Play saved USD poses without importing recorded-data readers."""

import json
from pathlib import Path

import numpy as np

DEMO_REPLAY = Path(__file__).resolve().parent / "assets/motions/excavator_demo.json"


def load_replay(path):
    """Read the cabin timeline used by both the small demo and full recorded replay plans."""
    frames = json.loads(Path(path).read_text(encoding="utf-8"))["views"]["cabin"]
    if not frames:
        raise ValueError("Replay must contain at least one cabin pose")
    previous = -1
    for frame in frames:
        stamp = frame["timestamp_ns"]
        if type(stamp) is not int or stamp <= previous:
            raise ValueError("Replay timestamps must be nonnegative integers in strictly increasing order")
        previous = stamp
        if not frame["transforms"]:
            raise ValueError("Replay poses must contain transforms")
        for matrix in frame["transforms"].values():
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("Replay transforms must be finite 4x4 USD matrices")
    return frames


def apply_transforms(stage, transforms, ops):
    """Apply a pose on the caller's edit layer, reusing USD transform operations."""
    from pxr import Gf, UsdGeom

    for path, matrix in transforms.items():
        if path not in ops:
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            if not xform:
                raise ValueError(f"Replay target does not exist: {path}")
            op = xform.AddTransformOp(opSuffix="recordedReplay")
            xform.SetXformOpOrder([op], xform.GetResetXformStack())
            ops[path] = op
        ops[path].Set(Gf.Matrix4d(*np.asarray(matrix).ravel()))
