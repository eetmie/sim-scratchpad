"""Load the exported robot description and derive self-filter boxes from its meshes."""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np


def read_urdf(ros2_dir):
    """urdf/excavator.urdf with its mesh paths pointed at this checkout.

    The exporter writes absolute mesh paths of the machine it ran on; the robot and other machines have their own.
    """
    with open(os.path.join(ros2_dir, "urdf", "excavator.urdf")) as f:
        return re.sub(r"file://[^\"]*/urdf/meshes/", f"file://{ros2_dir}/urdf/meshes/", f.read())


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quat_matrix(x, y, z, w):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def link_boxes(urdf_xml):
    """{link: (center, axes (rows), half extents)} from each link's visual OBJ meshes, in the link frame."""
    boxes = {}
    for link in ET.fromstring(urdf_xml).findall("link"):
        points = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            path = mesh.get("filename").removeprefix("file://")
            with open(path) as f:
                v = np.array([line.split()[1:4] for line in f if line.startswith("v ")], dtype=float)
            origin = visual.find("origin")
            xyz = [float(a) for a in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
            rpy = [float(a) for a in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
            points.append(v @ rpy_matrix(*rpy).T + xyz)
        if points:
            p = np.concatenate(points)
            center = p.mean(0)
            axes = np.linalg.eigh(np.cov((p - center).T))[1].T  # principal axes as rows
            local = (p - center) @ axes.T
            mid = (local.max(0) + local.min(0)) / 2
            boxes[link.get("name")] = (center + mid @ axes, axes, (local.max(0) - local.min(0)) / 2)
    return boxes


def box_params(urdf_xml, skip=("lower_carriage",)):
    """depth_to_cloud parameters: box_links and box_data (center xyz, axes as rows, half extents xyz per link).

    lower_carriage is skipped: it sits below the cameras' view and its box would only cut ground.
    """
    boxes = {k: v for k, v in link_boxes(urdf_xml).items() if k not in skip}
    data = [float(x) for c, axes, half in boxes.values() for x in (*c, *axes.ravel(), *half)]
    return {"box_links": list(boxes), "box_data": data}
