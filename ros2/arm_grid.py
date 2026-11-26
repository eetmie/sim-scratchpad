"""The robot's boom/arm/bucket self-filter (kaivuriprokkis arm_model.npz) as depth_to_cloud parameters.

arm_model.npz holds, per link, a distance grid (mm, uint8) to the link's mesh in a frame built from its pivot, the
next pivot (the bucket: its tip) and the upper-carriage lateral axis. kaivuriprokkis tools/arm_model.py builds it
from this URDF; the frame steps below are copied from it, so the grids land on the same URDF links here. The robot
stretches each grid to its FK link length; the sim's links are the URDF's, so no stretch.
Plain numpy, no ROS.
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

# grid link -> (URDF link the grid rides on, meshes on it, URDF joint at its pivot, next pivot joint; None: bucket tip)
LINKS = {
    1: ("lift_boom", ("lift_boom",), "revolute_lift", "revolute_tilt"),
    2: ("tilt_boom", ("tilt_boom",), "revolute_tilt", "revolute_tool"),
    3: ("tool_body", ("tool_body", "bucket"), "revolute_tool", None),
}
GRID_LINKS = tuple(v[0] for v in LINKS.values()) + ("bucket",)  # their boxes give way to the grids


def _rpy(r):
    x, y, z = r
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    return (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )


def _origin(e):
    o, M = e.find("origin"), np.eye(4)
    if o is not None:
        M[:3, :3] = _rpy([float(v) for v in o.get("rpy", "0 0 0").split()])
        M[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return M


def _load_obj(path):
    V, F = [], []
    with open(path) as lines:
        for line in lines:
            if line.startswith("v "):
                V.append(line.split()[1:4])
            elif line.startswith("f "):
                f = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                F += [(f[0], f[i], f[i + 1]) for i in range(1, len(f) - 1)]
    return np.asarray(V, float), np.asarray(F, int)


def _surface(V, F, spacing=0.0015, seed=0):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.random.default_rng(seed).poisson(area / spacing**2)
    i = np.repeat(np.arange(len(F)), n)
    r = np.random.default_rng(seed + 1).random((len(i), 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    return np.vstack([V, a[i] + r[:, :1] * (b[i] - a[i]) + r[:, 1:] * (c[i] - a[i])])


def _upper_frame(urdf_path):
    """Link poses and joint origins in the upper-carriage frame (joints at zero), and arm mesh surface points."""
    root = ET.parse(urdf_path).getroot()
    pose, joints = {"upper_carriage": np.eye(4)}, {}
    pending = list(root.findall("joint"))
    while pending:
        for j in list(pending):
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            if parent in pose:
                pose[child] = pose[parent] @ _origin(j)
                joints[j.get("name")] = pose[child][:3, 3]
                pending.remove(j)
            elif parent in ("world", "lower_carriage"):
                pending.remove(j)
    points = {}
    for link in root.findall("link"):
        name, vis = link.get("name"), link.find("visual")
        if name not in pose or vis is None or not any(name in v[1] for v in LINKS.values()):
            continue
        mesh = os.path.join(
            os.path.dirname(urdf_path), "meshes", os.path.basename(vis.find("geometry/mesh").get("filename"))
        )
        V, F = _load_obj(mesh)
        M = pose[name] @ _origin(vis)
        points[name] = _surface(V, F) @ M[:3, :3].T + M[:3, 3]
    return pose, joints, points


def _link_frame(pivot, nxt, lateral):
    e1 = nxt - pivot
    e1 = e1 - (e1 @ lateral) * lateral
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, lateral, np.cross(e1, lateral)])


def arm_params(urdf_path, npz_path, margins_m, grid_file):
    """depth_to_cloud parameters arm_links / arm_data / arm_grid_file; writes the uint8 grids to grid_file.

    arm_data per link: A (3x3 rows), b (3): grid-frame point q = A p_link + b; lo (3), shape (3), voxel, margin_m.
    """
    z = np.load(npz_path)
    voxel = float(z["voxel"])
    pose, joints, points = _upper_frame(urdf_path)
    lateral = np.array([0, 1.0, 0])
    links, data, grids = [], [], []
    for k, (link, meshes, pivot_joint, next_joint) in LINKS.items():
        pivot = joints[pivot_joint]
        if next_joint is None:  # bucket tip: the lip point farthest from the hinge, mid-width (as arm_model.py)
            P = np.vstack([points[n] for n in meshes])
            mid = np.abs(P[:, 1] - np.median(P[:, 1])) < 0.004
            d = np.linalg.norm((P - pivot)[:, [0, 2]], axis=1)
            d[~mid] = 0
            nxt = P[np.argmax(d)]
        else:
            nxt = joints[next_joint]
        E = _link_frame(pivot, nxt, lateral)
        length = float((nxt - pivot) @ E[0])
        if abs(length - float(z[f"link{k}_length"])) > 1e-4:
            raise ValueError(
                f"{npz_path}: link {k} length {z[f'link{k}_length']:.4f} != URDF {length:.4f}; "
                "rebuild it with kaivuriprokkis tools/arm_model.py from this URDF"
            )
        R, t = pose[link][:3, :3], pose[link][:3, 3]
        grid = np.ascontiguousarray(z[f"link{k}_mm"])
        links.append(link)
        data += [*(E @ R).ravel(), *(E @ (t - pivot)), *z[f"link{k}_lo"], *grid.shape, voxel, float(margins_m[k - 1])]
        grids.append(grid)
    os.makedirs(os.path.dirname(grid_file), exist_ok=True)
    with open(grid_file, "wb") as f:
        for g in grids:
            f.write(g.tobytes())
    return {"arm_links": links, "arm_data": [float(x) for x in data], "arm_grid_file": grid_file}


if __name__ == "__main__":  # self-check: the grid frames put the URDF's own arm surface inside the margins
    here = os.path.dirname(os.path.abspath(__file__))
    urdf = os.path.join(here, "urdf", "excavator.urdf")
    out = os.path.join(here, "ws", "arm_grids.bin")
    p = arm_params(urdf, os.path.join(here, "arm_model.npz"), (0.09, 0.045, 0.10), out)
    blob = np.fromfile(out, np.uint8)
    pose, _, points = _upper_frame(urdf)
    offset = 0
    for i, link in enumerate(p["arm_links"]):
        d = p["arm_data"][20 * i : 20 * i + 20]
        A, b, lo, shape, voxel = (
            np.reshape(d[:9], (3, 3)),
            np.array(d[9:12]),
            np.array(d[12:15]),
            tuple(int(s) for s in d[15:18]),
            d[18],
        )
        grid = blob[offset : offset + np.prod(shape)].reshape(shape)
        offset += grid.size
        meshes = LINKS[i + 1][1]
        P = np.vstack([points[n] for n in meshes])
        M = pose[link]
        p_link = (P - M[:3, 3]) @ M[:3, :3]  # upper -> link frame
        ijk = np.floor((p_link @ A.T + b - lo) / voxel).astype(int)
        ok = ((ijk >= 0) & (ijk < shape)).all(1)
        on = grid[tuple(ijk[ok].T)] <= 4  # surface points sit in or next to an occupied voxel
        print(f"{link}: {ok.mean():.3f} of surface in grid, {on.mean():.3f} within 4 mm")
        assert ok.mean() > 0.999 and on.mean() > 0.999
    print("ok")
