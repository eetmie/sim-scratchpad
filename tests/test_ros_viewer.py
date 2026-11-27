"""Wire-format geometry, camera and clock contracts, runnable without ROS or a renderer."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from depth_mapping import level_points, level_pose
from ros2.map_contract import FrameClock, depth_millimetres, grid_points, height_image, height_metres, image_calibration
from ros2.viewer_scene import joint_positions
from scipy.spatial.transform import Rotation


def test_map_payloads_keep_unknown_cells_and_ground_reference():
    mm = np.array([[-32768, -120], [0, 120]], dtype=np.int16)
    height = height_metres(mm)
    assert np.isnan(height[0, 0])
    np.testing.assert_allclose(height.ravel()[1:], [-0.12, 0, 0.12], atol=1e-7)
    np.testing.assert_array_equal(height_image(mm), [[0, 16], [128, 240]])
    points = grid_points(height, [0.1, 0.3], [-0.1, 0.1], 0.1)
    np.testing.assert_allclose(points, [[0.25, -0.05, -0.12], [0.15, 0.05, 0], [0.15, -0.05, 0.12]], atol=1e-7)
    assert points.dtype == np.dtype("<f4") and points.flags.c_contiguous
    pose = level_pose(np.array([2, 3, 99]), np.pi / 2, -0.0245)
    world = points @ pose[:3, :3].T + pose[:3, 3]
    np.testing.assert_allclose(level_points(world, np.array([2, 3, 99]), np.pi / 2, -0.0245), points, atol=1e-7)
    assert len(grid_points(np.full((2, 2), np.nan), [0, 1], [0, 1], 0.5)) == 0


def test_optional_depth_converts_scale_and_reverses_recorded_rotation():
    sensor = {"rotation_deg": 180, "depth_scale_m": 0.0001}
    recorded = np.array([[1000, 2000], [0, 6000]], dtype=np.uint16)
    converted = depth_millimetres(recorded, sensor)
    np.testing.assert_array_equal(converted, [[600, 0], [200, 100]])
    assert converted.dtype == np.dtype("<u2") and converted.flags.c_contiguous


def test_rotated_ir_camera_info_and_tf_project_the_same_pixel():
    info = {
        "image_rotation_deg": 180,
        "intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 390.0,
            "fy": 385.0,
            "cx": 325.5,
            "cy": 236.2,
            "coeffs": [0, 0, 0.01, -0.02, 0],
        },
    }
    intr, roll = image_calibration(info)
    point = np.array([0.1, -0.2, 0.6])
    original = np.array([390 * point[0] / point[2] + 325.5, 385 * point[1] / point[2] + 236.2])
    rotated = roll[:3, :3].T @ point
    projected = [intr["fx"] * rotated[0] / rotated[2] + intr["cx"], intr["fy"] * rotated[1] / rotated[2] + intr["cy"]]
    np.testing.assert_allclose(projected, [639, 479] - original)
    assert intr["coeffs"][2:4] == [-0.01, 0.02]
    assert info["intrinsics"]["cx"] == 325.5


def test_replay_clock_preserves_exposure_intervals_despite_slow_rendering_and_loops():
    source = [123_000_000_000, 123_200_000_000, 123_500_000_000]
    clock = FrameClock("auto", 5, source)
    stamps = [clock.stamp(i, 999_000_000_000 + i * 4_000_000_000) for i in range(6)]
    assert stamps == [1_000_000_000, 1_200_000_000, 1_500_000_000, 1_750_000_000, 1_950_000_000, 2_250_000_000]
    np.testing.assert_array_equal(np.diff(stamps[:3]), np.diff(source))
    assert clock.mode == "replay"
    with pytest.raises(ValueError, match="strictly"):
        FrameClock("auto", 5, [10, 10])


def test_fixed_clock_and_live_wall_clock_are_explicitly_different():
    fixed, wall = FrameClock("auto", 5), FrameClock("wall", 5)
    assert fixed.stamp(0, 500_000_000_000) == wall.stamp(0, 500_000_000_000) == 1_000_000_000
    assert fixed.stamp(1, 503_000_000_000) == 1_200_000_000
    assert wall.stamp(1, 503_000_000_000) == 4_000_000_000
    for invalid in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            FrameClock("auto", invalid)


def test_joint_states_include_urdf_zero_offsets_under_a_rotated_base():
    joint = ET.fromstring("""<joint name="lift" type="revolute">
        <parent link="base"/><child link="boom"/><axis xyz="0 1 0"/>
        <origin xyz="0 0 0" rpy="0 -0.455 0"/></joint>""")
    parent = np.eye(4)
    parent[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 1.1]).as_matrix()
    child = parent.copy()
    child[:3, :3] = parent[:3, :3] @ Rotation.from_euler("y", -0.455 + 0.37).as_matrix()
    assert joint_positions({"base": parent, "boom": child}, [joint])["lift"] == pytest.approx(0.37)


def test_fastdem_default_follows_the_native_build_directory(tmp_path, monkeypatch):
    import os

    import robot_mapping

    build, binary = ("build-mingw", "fastdem_replay.exe") if os.name == "nt" else ("build", "fastdem_replay")
    adapter = tmp_path / "logs" / "fastdem" / build / binary
    adapter.parent.mkdir(parents=True)
    adapter.touch()
    monkeypatch.setattr(robot_mapping, "ROOT", tmp_path)
    assert robot_mapping.fastdem_binary() == adapter.resolve()


def test_usd_snapshot_has_one_tf_parent_per_link_and_reconstructs_live_poses():
    from pathlib import Path

    from ros2.viewer_scene import ViewerScene

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(Path(__file__).resolve().parents[1] / "demo_env.usd"), load=Usd.Stage.LoadNone)
    scene = ViewerScene(stage)
    front, surround = level_pose(np.array([0.1, 0.2, 0]), 0.7, -0.0245), level_pose(np.zeros(3), 0.2, -0.0245)
    arrays = {
        "front_to_world": front,
        "surround_to_world": surround,
        "optical_to_world_cabin": np.eye(4),
        "optical_to_world_arm": np.eye(4),
    }
    snapshot = scene.sample(arrays)
    parents = [child for _, child, _ in snapshot["transforms"]]
    assert len(parents) == len(set(parents))
    assert len(snapshot["joints"]) == 4
    assert all(np.isfinite(v) for v in snapshot["joints"].values())
    world = {"world": np.eye(4)}
    for parent, child, matrix in snapshot["transforms"]:
        world[child] = world[parent] @ matrix
    cache = UsdGeom.XformCache()
    for link in scene.links:
        expected = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/excavator_bucket/{link}"))).T
        np.testing.assert_allclose(world[link], expected, atol=1e-10)
    np.testing.assert_allclose(world["excavator_front_map"], front)
    np.testing.assert_allclose(world["excavator_surround_map"], surround)
    np.testing.assert_allclose(world["arm_camera"], np.eye(4), atol=1e-10)
    scene.close()
