"""Export the excavator in demo_env.usd to URDF + meshes for RViz (robot_state_publisher).

    uv run --extra isaacsim python scripts/sim-scratchpad/ros2/export_urdf.py

Writes ros2/urdf/excavator.urdf (mesh paths are absolute file:// URIs, so no ROS package is needed).
Joint origins and axes come from the USD joint frames, so URDF joint angles equal the sim's /joint_states.
A fixed "world" root link sits at the excavator's stage pose, so RViz's fixed frame "world" matches the sim.
"""

import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from scipy.spatial.transform import Rotation  # noqa: E402

import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
USD = os.path.join(HERE, "..", "demo_env.usd")
OUT_DIR = os.path.join(HERE, "urdf")
URDF = os.path.join(OUT_DIR, "excavator.urdf")
ROBOT = "/World/excavator_bucket"
ROOT_LINK = "lower_carriage"
# The exporter needs a finite velocity limit per joint; use the teleop rate speeds (deg/s -> rad/s).
# Authored on the in-memory stage only, demo_env.usd is not changed.
VELOCITY_LIMITS_DEG_S = {"revolute_carriage": 15.0, "revolute_tilt": 12.5, "revolute_tool": 20.0, "revolute_lift": 10.0}

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.exporter.urdf", True)
from isaacsim.asset.exporter.urdf import UsdToUrdfConverter  # noqa: E402

omni.usd.get_context().open_stage(os.path.abspath(USD))
for _ in range(10):
    app.update()
stage = omni.usd.get_context().get_stage()
for name, speed in VELOCITY_LIMITS_DEG_S.items():
    joint = stage.GetPrimAtPath(f"{ROBOT}/Joints/{name}")
    if not joint.GetAttribute("urdf:limit:velocity").HasAuthoredValue():
        joint.CreateAttribute("urdf:limit:velocity", Sdf.ValueTypeNames.Float).Set(math.radians(speed))

# The exporter numbers meshes after existing files, so old exports would pile up.
shutil.rmtree(os.path.join(OUT_DIR, "meshes"), ignore_errors=True)
os.makedirs(OUT_DIR, exist_ok=True)
converter = UsdToUrdfConverter(
    stage=stage,
    root_prim_path=ROBOT,
    mesh_dir_name="meshes",
    mesh_path_prefix=f"file://{OUT_DIR}/meshes/",
)
converter.convert(URDF)
# Some exported OBJs name an .mtl the exporter never writes; RViz then refuses the mesh.
for obj in Path(OUT_DIR, "meshes").glob("*.obj"):
    lines = obj.read_text().splitlines(keepends=True)
    kept = [line for line in lines if not (line.startswith("mtllib ") and not (obj.parent / line.split()[1]).exists())]
    if len(kept) != len(lines):
        obj.write_text("".join(kept))


def rotation(gf_quat):
    return Rotation.from_quat([*gf_quat.GetImaginary(), gf_quat.GetReal()])


def set_origin(element, xyz, rot):
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    origin.set("xyz", " ".join(f"{v:.9g}" for v in xyz))
    origin.set("rpy", " ".join(f"{v:.9g}" for v in rot.as_euler("xyz")))


tree = ET.parse(URDF, ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
robot = tree.getroot()
# The exporter derives joint origins from the authored link poses, which puts joint zero at the authored
# pose. PhysX (and /joint_states) measure angles between the joint frames instead, so rebuild origin and
# axis from those: child = parent * (localPos0, localRot0) * R(axis, q) * (localPos1, localRot1)^-1.
for element in robot.findall("joint"):
    prim = stage.GetPrimAtPath(f"{ROBOT}/Joints/{element.get('name')}")
    if not prim.IsA(UsdPhysics.Joint):
        continue
    joint = UsdPhysics.Joint(prim)
    rot0, rot1 = rotation(joint.GetLocalRot0Attr().Get()), rotation(joint.GetLocalRot1Attr().Get())
    pos0, pos1 = np.array(joint.GetLocalPos0Attr().Get()), np.array(joint.GetLocalPos1Attr().Get())
    set_origin(element, pos0 - (rot0 * rot1.inv()).apply(pos1), rot0 * rot1.inv())
    if prim.IsA(UsdPhysics.RevoluteJoint):
        # URDF rotates about the child link origin, so the child's joint frame must sit there.
        assert np.linalg.norm(pos1) < 1e-6, f"{prim.GetPath()}: localPos1 {pos1} is not at the child link origin"
        axis = rot1.apply(np.eye(3)["XYZ".index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())])
        element.find("axis").set("xyz", " ".join(f"{v:.9g}" for v in axis))

# Link frames are now the USD body frames, but the exporter writes meshes in its own link frames.
# Find that frame per mesh with an exact rigid fit (same vertices, same order) and set it as origin.
cache = UsdGeom.XformCache(Usd.TimeCode.Default())


def obj_frame_in_body(obj_uri, mesh_path, body_path):
    """4x4 transform from the OBJ's coordinates to the body frame."""
    lines = Path(obj_uri.removeprefix("file://")).read_text().splitlines()
    obj = np.array([line.split()[1:4] for line in lines if line.startswith("v ")], dtype=float)
    mesh = stage.GetPrimAtPath(mesh_path)
    to_body = np.array(cache.GetLocalToWorldTransform(mesh)) @ np.linalg.inv(
        np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)))
    )  # USD matrices are row-vector
    points = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=float)
    body = (np.c_[points, np.ones(len(points))] @ to_body)[:, :3]
    assert len(obj) == len(body), f"{mesh_path}: {len(obj)} OBJ vertices vs {len(body)} USD points"
    u, _, vt = np.linalg.svd((obj - obj.mean(0)).T @ (body - body.mean(0)))  # Kabsch: body = r @ obj + t
    r = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, body.mean(0) - r @ obj.mean(0)
    residual = np.abs(obj @ r.T + t[:3, 3] - body).max()
    assert residual < 1e-5, f"{mesh_path}: OBJ is not a rigid copy of the USD mesh (off by {residual * 1000:.3f} mm)"
    return t


def origin_matrix(element):
    t = np.eye(4)
    origin = element.find("origin")
    if origin is not None:
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        t[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return t


for link in robot.findall("link"):
    body_path = f"{ROBOT}/{link.get('name')}"
    for tag, scope in (("visual", "visuals"), ("collision", "colliders")):
        for element in link.findall(tag):
            uri = element.find("geometry/mesh").get("filename")
            t = obj_frame_in_body(uri, f"{body_path}/{scope}/Mesh", body_path) @ origin_matrix(element)
            set_origin(element, t[:3, 3], Rotation.from_matrix(t[:3, :3]))
    # Inertials are already in the body frame (the exporter copies the USD center of mass).

# world -> root link at the stage pose, so RViz's fixed frame "world" matches the sim.
assert robot.find(f"link[@name='{ROOT_LINK}']") is not None, f"no link {ROOT_LINK!r}; update ROOT_LINK"
pose = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(stage.GetPrimAtPath(f"{ROBOT}/{ROOT_LINK}"))
world_joint = ET.Element("joint", name="world_fixed", type="fixed")
ET.SubElement(world_joint, "parent", link="world")
ET.SubElement(world_joint, "child", link=ROOT_LINK)
set_origin(world_joint, pose.ExtractTranslation(), rotation(pose.ExtractRotationQuat()))
robot.insert(0, world_joint)
robot.insert(0, ET.Element("link", name="world"))
ET.indent(tree)
tree.write(URDF, encoding="utf-8", xml_declaration=True)
print(f"[export_urdf] wrote {URDF}; check it with check_urdf_fk.py")
app.close()
