"""Remembered lidar height map; pure NumPy, so any engine (RTX, Newton, soil later) can feed it.

Same algorithm and constants as ../lidar_demo_script.py: a fixed WORLD-space crop, highest
return per cell, cells seen in a scan replace their memory, and rays that pass below a
remembered height clear it.
"""

import numpy as np

HEIGHTMAP_SIZE = 64
HEIGHTMAP_X_M = (0.16, 0.80)
HEIGHTMAP_Y_M = (-0.32, 0.32)
GROUND_Z_M = 0.0
HEIGHTMAP_HEIGHT_M = (-0.15, 0.15)  # Fixed brightness scale: ground is middle gray.
# A remembered cell is cleared when a ray to a farther hit passes this far below its stored height.
CLEAR_MARGIN_M = 0.02


def new_heightmap_memory():
    """Empty persistent map: the scan fields plus when each cell was last seen."""
    memory = rasterize_heightmap(np.empty((0, 3)))
    memory["last_seen_ns"] = np.zeros(memory["valid"].shape, dtype=np.int64)
    memory["fresh"] = np.zeros(memory["valid"].shape, dtype=bool)
    memory["stamp_ns"] = 0
    return memory


def merge_heightmap(memory, scan_map, stamp_ns):
    """Cells seen in this scan replace the old value (so digging lowers them); unseen cells keep theirs."""
    seen = scan_map["valid"]
    for key in ("height_m", "gray_u8"):
        memory[key][seen] = scan_map[key][seen]
    memory["valid"] |= seen
    memory["last_seen_ns"][seen] = stamp_ns
    memory["fresh"] = seen
    memory["counts"] = scan_map["counts"]
    memory["stamp_ns"] = stamp_ns
    return memory


def world_to_cells(world):
    """Flat cell index (row-major, row 0 = +X edge, column 0 = +Y edge) for points inside the crop."""
    size = HEIGHTMAP_SIZE
    x0, x1 = HEIGHTMAP_X_M
    y0, y1 = HEIGHTMAP_Y_M
    inside = (
        np.isfinite(world).all(axis=-1)
        & (world[..., 0] >= x0)
        & (world[..., 0] < x1)
        & (world[..., 1] >= y0)
        & (world[..., 1] < y1)
    )
    points = world[inside]
    ix = np.minimum(((points[:, 0] - x0) * size / (x1 - x0)).astype(np.int64), size - 1)
    iy = np.minimum(((points[:, 1] - y0) * size / (y1 - y0)).astype(np.int64), size - 1)
    return (size - 1 - ix) * size + (size - 1 - iy), inside


def clear_seen_through(memory, origin, hits):
    """Forget remembered cells that a lidar ray just passed through, below their stored height.

    A surface there would have blocked the ray, so the stored value is stale. Cells hidden
    behind the bucket keep their value: no terrain ray crosses them. Returns how many cells were cleared.
    """
    if not len(hits):
        return 0
    step = (HEIGHTMAP_X_M[1] - HEIGHTMAP_X_M[0]) / HEIGHTMAP_SIZE
    rays = hits - origin
    reach = np.linalg.norm(rays[:, :2], axis=1)
    # Sample every cell, stopping two cells short of the hit so the hit's own cell is never cleared.
    t = np.arange(1, int(reach.max(initial=0) / step) + 1) * step / np.maximum(reach, 1e-9)[:, None]
    along = t < 1 - 2 * step / np.maximum(reach, 1e-9)[:, None]
    samples = origin + t[along][:, None] * rays[np.nonzero(along)[0]]
    cells, inside = world_to_cells(samples)
    stored = memory["height_m"].ravel()[cells]
    stale = np.unique(cells[stored > samples[inside, 2] - GROUND_Z_M + CLEAR_MARGIN_M])
    for key, empty in (("height_m", np.nan), ("valid", False), ("gray_u8", 0), ("last_seen_ns", 0)):
        memory[key].ravel()[stale] = empty
    return len(stale)


def rasterize_heightmap(world):
    """Highest return per XY cell within ONE scan; merge_heightmap adds the memory across scans.

    Row zero is +world X (far edge); column zero is +world Y (left edge).
    Unobserved cells are NaN in height_m, false in valid, and zero in gray_u8.
    """
    size = HEIGHTMAP_SIZE
    cells, inside = world_to_cells(world)
    points = world[inside]
    counts = np.bincount(cells, minlength=size * size).reshape(size, size)
    height = np.full(size * size, -np.inf, dtype=np.float32)
    np.maximum.at(height, cells, points[:, 2] - GROUND_Z_M)
    height = height.reshape(size, size)
    valid = counts > 0
    height[~valid] = np.nan
    gray = np.zeros((size, size), dtype=np.uint8)
    low, high = HEIGHTMAP_HEIGHT_M
    gray[valid] = np.rint(255 * np.clip((height[valid] - low) / (high - low), 0, 1)).astype(np.uint8)
    return {"height_m": height, "valid": valid, "gray_u8": gray, "counts": counts}


def update_heightmap(memory, origin, terrain_hits, stamp_ns):
    """One scan: clear what the rays saw through, then merge this scan's cells. Returns (memory, cleared)."""
    cleared = clear_seen_through(memory, origin, terrain_hits)
    return merge_heightmap(memory, rasterize_heightmap(terrain_hits), stamp_ns), cleared
