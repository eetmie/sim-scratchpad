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


def test_cloud_follows_each_map_and_depth_payloads_follow_their_subscribers():
    from unittest.mock import Mock, patch

    from ros2.viewer_bridge import ViewerRosBridge

    bridge = ViewerRosBridge.__new__(ViewerRosBridge)
    bridge.cloud_hz, bridge.last_cloud_ns = None, None
    bridge.image, bridge.header, bridge.node = Mock(), Mock(), Mock()
    bridge.types = {"PointField": Mock(), "PointCloud2": Mock()}
    cloud, depth = Mock(), Mock()
    bridge.publishers = {"cloud": cloud, "depth_arm": depth, "depth_info_arm": Mock()}
    bridge.publishers["depth_info_arm"].get_subscription_count.return_value = 0
    bridge.profile = {
        "height_color_limit_m": 0.12,
        "surround": {"xy_m": [0, 1], "cell_m": 0.5},
        "sensors": {"arm": {"rotation_deg": 0, "depth_scale_m": 0.001}},
    }
    grid = np.array([[0, 5], [-32768, 10]], dtype=np.int16)
    arrays = dict(front_mm=grid, surround_mm=grid, surround_age_frames=grid, depth_arm_z16=np.ones_like(grid))
    with patch("ros2.viewer_bridge.grid_points", wraps=grid_points) as points:
        for stamp, subscribers, enabled in [(1, 0, True), (2, 1, False), (3, 1, True), (4, 1, True)]:
            cloud.get_subscription_count.return_value = subscribers
            depth.get_subscription_count.return_value = int(stamp == 2)
            bridge.node.get_parameter.return_value.value = enabled
            bridge.publish_maps(arrays, stamp)
        assert points.call_count == cloud.publish.call_count == 2
    assert [c.args[0] for c in bridge.header.call_args_list] == [3, 4]
    assert [c.args[3] for c in bridge.image.call_args_list if c.args[0] == "depth_arm"] == [2]
    assert [c.args[3] for c in bridge.image.call_args_list if c.args[0] == "surround"] == [1, 2, 3, 4]


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


def test_headless_capture_overlaps_mapping_and_publishes_the_acquisition_snapshot_in_order():
    from threading import Event
    from time import monotonic
    from types import SimpleNamespace
    from unittest.mock import Mock

    from launch_3dgs import RosFrames

    second_capture = Event()
    mapping = Mock()
    mapping.ir_views = {}
    images = [np.full((2, 2), i, np.uint8) for i in (1, 2)]
    frames = [SimpleNamespace(started=monotonic(), arrays={"capture_host_monotonic_ns": i}, timings={}) for i in (1, 2)]

    def capture():
        frame = frames.pop(0)
        i = frame.arrays["capture_host_monotonic_ns"]
        mapping.ir_views["arm"] = images[i - 1]
        if i == 2:
            second_capture.set()
        return frame

    def map_frame(frame):
        assert second_capture.wait(2)
        frame.arrays["front_mm"] = frame.arrays["capture_host_monotonic_ns"]

    mapping.capture_robot.side_effect = capture
    mapping.map_robot.side_effect = map_frame
    scene = Mock()
    scene.sample.side_effect = lambda arrays, views: dict(views)
    bridge = Mock()
    stream = RosFrames(mapping, scene, bridge, FrameClock("auto", 5, [10, 20]), headless=True)
    try:
        assert stream.update(0, 10)
        assert stream.update(0, 20)
    finally:
        stream.close()
    assert len(bridge.publish.call_args_list) == 2
    assert mapping.record_robot_frame.call_count == 2
    for i, call in enumerate(bridge.publish.call_args_list):
        arrays, snapshot = call.args
        assert arrays["front_mm"] == i + 1 and snapshot["arm"] is images[i]
        assert arrays["ros_stamp_ns"] == call.kwargs["stamp_ns"] == 1_000_000_000 + 10 * i
        assert call.kwargs["source_ns"] == 10 * (i + 1)


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
    from types import SimpleNamespace

    from camera_calibration.usd_cameras import cameras, describe
    from ir_views import scaled_camera
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
    rig = cameras(stage)
    previews = SimpleNamespace(
        products={name: (scaled_camera(describe(camera), 224), None, None) for name, camera in rig.items()},
        images={name: np.zeros((168, 224), dtype=np.uint8) for name in rig},
    )
    snapshot = scene.sample(arrays, previews)
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
    for name, camera in rig.items():
        pixels, intr = snapshot["images"][name]
        assert pixels is previews.images[name] and (intr["width"], intr["height"]) == (224, 168)
        info = describe(camera)
        _, roll = image_calibration(info)
        pose = info["optical_to_world"]
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix()
        expected[:3, 3] = pose["translation_m"]
        np.testing.assert_allclose(world[f"{name}_ir_optical"], expected @ roll, atol=1e-7)


def test_live_ir_and_maps_share_acquisition_stamps_and_skip_missed_slots():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from launch_3dgs import LiveSensors

    mapping, ros = Mock(), Mock()
    mapping.capture_hz = 5
    mapping.robot_policy.policy = {"capture": {"ir_fps": 30}}
    ros.clock = FrameClock("wall", 5)
    ros.count = 0
    stream = LiveSensors(mapping, ros)
    try:
        # Two camera slots are missed halfway through the run.
        slots = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12]
        for slot in slots:
            now = 10 + slot / 30 + 1e-6
            with patch("launch_3dgs.time.monotonic", return_value=now), patch("launch_3dgs.time.sleep"):
                stream.pace()
            arrays = {"capture_host_monotonic_ns": round(now * 1e9)}
            mapping.capture_clock = arrays
            mapping.capture_robot.return_value = SimpleNamespace(arrays=arrays, started=now, timings={})
            stream.update(0, slot)
    finally:
        stream.close()
    exposures = [c.kwargs["stamp_ns"] for c in ros.bridge.publish_exposure.call_args_list]
    maps = [c.args[1] for c in ros.bridge.publish_maps.call_args_list]
    assert len(exposures) == len(slots)
    assert maps == [exposures[i] for i in [0, 6, 10]]
    assert mapping.capture.call_count == 8
    assert all(c.kwargs == {"with_depth": False} for c in mapping.capture.call_args_list)
    assert mapping.record_robot_frame.call_count == mapping.show_robot_maps.call_count == 3
