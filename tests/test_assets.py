"""Shared URDF assets must work from a fresh checkout on any machine."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ros2.robot_boxes import link_boxes, read_urdf

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/urdf/excavator.urdf"


def test_urdf_uses_existing_relative_meshes_and_loads_filter_geometry():
    root = ET.parse(URDF).getroot()
    meshes = root.findall(".//mesh")
    assert len(meshes) == 12
    for mesh in meshes:
        filename = mesh.get("filename")
        assert filename.startswith("meshes/")
        assert (URDF.parent / filename).is_file()
    boxes = link_boxes(read_urdf(ROOT / "ros2"))
    assert set(boxes) == {link.get("name") for link in root.findall("link") if link.find("visual") is not None}
    for center, axes, half in boxes.values():
        assert np.isfinite(center).all() and np.isfinite(axes).all()
        assert (half > 0).all()


def test_ros_mesh_uris_follow_checkout_with_spaces_and_different_cwd(tmp_path, monkeypatch):
    checkout = tmp_path / "another checkout"
    destination = checkout / "assets/urdf"
    destination.mkdir(parents=True)
    shutil.copyfile(URDF, destination / URDF.name)
    monkeypatch.chdir(tmp_path)
    xml = read_urdf(checkout / "ros2")
    for mesh in ET.fromstring(xml).findall(".//mesh"):
        name = mesh.get("filename").rsplit("/", 1)[-1]
        assert mesh.get("filename") == (destination / "meshes" / name).as_uri()
    assert "%20" in xml
