"""Check the exported URDF against the sim: what RViz draws vs where the sim's meshes are, same joint angles.

    uv run --extra isaacsim python scripts/sim-scratchpad/ros2/check_urdf_fk.py [path/to/robot.urdf]

Places each link's URDF visual mesh with forward kinematics at the sim's joint angles, and measures how far
its vertices are from the sim's visual mesh (PhysX body pose). Non-zero exit if any link is off by > 2 mm.
"""

import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from omni.physx import get_physx_interface  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = "/World/excavator_bucket"


def transform(xyz, rot):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot.as_matrix(), xyz
    return t


def origin_transform(element):
    o = element.find("origin")
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
    return transform(xyz, Rotation.from_euler("xyz", rpy))


def apply(t, points):
    return points @ t[:3, :3].T + t[:3, 3]


def urdf_fk(joints, q):
    """World pose of every link for joint angles q (name -> rad)."""
    poses = {"world": np.eye(4)}
    while len(poses) <= len(joints):
        for j in joints:
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            if parent not in poses or child in poses:
                continue
            t = poses[parent] @ origin_transform(j)
            if j.get("type") in ("revolute", "continuous"):
                axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                t = t @ transform([0, 0, 0], Rotation.from_rotvec(axis * q[j.get("name")]))
            poses[child] = t
    return poses


omni.usd.get_context().open_stage(os.path.join(HERE, "..", "demo_env.usd"))
for _ in range(20):
    app.update()
omni.timeline.get_timeline_interface().play()
for _ in range(120):  # let the drives settle away from zero
    app.update()
robot = Articulation(ROBOT)
q = dict(zip(robot.dof_names, robot.get_dof_positions().numpy()[0].tolist()))
print("joint angles (deg):", {k: round(math.degrees(v), 2) for k, v in q.items()})
urdf = ET.parse(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "urdf", "excavator.urdf")).getroot()
poses = urdf_fk(urdf.findall("joint"), q)
stage = omni.usd.get_context().get_stage()
cache = UsdGeom.XformCache(Usd.TimeCode.Default())  # authored (USD) poses; PhysX does not write back to USD
worst = 0.0
for link in urdf.findall("link"):
    visual = link.find("visual")
    if visual is None:
        continue
    name = link.get("name")
    # URDF side: OBJ vertices -> link frame -> world.
    obj = visual.find("geometry/mesh").get("filename").removeprefix("file://")
    lines = Path(obj).read_text().splitlines()
    verts = np.array([line.split()[1:4] for line in lines if line.startswith("v ")], dtype=float)
    urdf_world = apply(poses[name] @ origin_transform(visual), verts)
    # Sim side: mesh points -> authored body frame -> PhysX body pose (USD matrices are row-vector).
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{ROBOT}/{name}/visuals/Mesh"))
    mesh_to_body = np.array(cache.GetLocalToWorldTransform(mesh.GetPrim())) @ np.linalg.inv(
        np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{ROBOT}/{name}")))
    )
    points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    in_body = (np.c_[points, np.ones(len(points))] @ mesh_to_body)[:, :3]
    sim = get_physx_interface().get_rigidbody_transformation(f"{ROBOT}/{name}")
    sim_world = apply(transform(sim["position"], Rotation.from_quat(sim["rotation"])), in_body)
    sample = urdf_world[:: max(1, len(urdf_world) // 5000)]
    error = cKDTree(sim_world).query(sample)[0].max() * 1000
    worst = max(worst, error)
    print(f"{name:15s} max visual vertex error {error:8.2f} mm")
print("PASS" if worst <= 2.0 else "FAIL")
app.close()
sys.exit(0 if worst <= 2.0 else 1)
