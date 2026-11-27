"""Read paused USD/replay poses and optional calibrated IR images. No ROS imports."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ros2.map_contract import FRONT_FRAME, SURROUND_FRAME, image_calibration
from ros2.robot_boxes import read_urdf
from scipy.spatial.transform import Rotation


def joint_positions(links, joints):
    """URDF angles from actual body rotations, including kinematic replay's direct link overrides."""
    positions = {}
    for joint in joints:
        if joint.get("type") not in ("revolute", "continuous"):
            continue
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        origin = joint.find("origin")
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
        relative = Rotation.from_euler("xyz", rpy).as_matrix().T @ links[parent][:3, :3].T @ links[child][:3, :3]
        q = Rotation.from_matrix(relative).as_quat()
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
        angle = 2 * np.arctan2(q[:3] @ (axis / np.linalg.norm(axis)), q[3])
        positions[joint.get("name")] = float((angle + np.pi) % (2 * np.pi) - np.pi)
    return positions


class ViewerScene:
    def __init__(self, stage):
        from pxr import UsdGeom

        self.stage = stage
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
            raise ValueError("The ROS viewer expects the current metre-scale scene")
        root = ET.fromstring(read_urdf(Path(__file__).resolve().parent.as_posix()))
        self.joints = root.findall("joint")
        self.links = [link.get("name") for link in root.findall("link") if link.get("name") != "world"]

    def sample(self, arrays, ir_views=None):
        from pxr import UsdGeom

        cache = UsdGeom.XformCache()
        links = {"world": np.eye(4)}
        for name in self.links:
            prim = self.stage.GetPrimAtPath(f"/World/excavator_bucket/{name}")
            if not prim:
                raise ValueError(f"URDF link is missing in USD: {name}")
            links[name] = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        transforms = []
        for joint in self.joints:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            transforms.append((parent, child, np.linalg.inv(links[parent]) @ links[child]))
        if "front_to_world" in arrays:
            transforms.extend(
                [
                    ("world", FRONT_FRAME, arrays["front_to_world"]),
                    ("world", SURROUND_FRAME, arrays["surround_to_world"]),
                ]
            )
            for name, parent in (("cabin", "upper_carriage"), ("arm", "tilt_boom")):
                transforms.append(
                    (parent, f"{name}_camera", np.linalg.inv(links[parent]) @ arrays[f"optical_to_world_{name}"])
                )
        images = {}
        if ir_views is not None:
            for name, pixels in ir_views.images.items():
                info = ir_views.products[name][0]
                intr, roll = image_calibration(info)
                pose = np.eye(4)
                mount = info["mount"]
                pose[:3, :3] = Rotation.from_quat(mount["quaternion_xyzw"]).as_matrix()
                pose[:3, 3] = mount["translation_m"]
                parent = info["parent"].rsplit("/", 1)[-1]
                transforms.append((parent, f"{name}_ir_optical", pose @ roll))
                images[name] = (pixels, intr)
        return {"transforms": transforms, "joints": joint_positions(links, self.joints), "images": images}
