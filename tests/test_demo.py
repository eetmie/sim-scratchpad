"""Portable profile and motion checks; no recordings or saved comparison outputs required."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from heightmap_config import DEFAULT_CONFIG, STREAMS, load_profile
from launch_3dgs import parse_arguments
from scene_motion import DEMO_REPLAY, apply_transforms, load_replay

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("KAIVURI_REPO", Path.home() / "Documents/GitHub/kaivuriprokkis"))


def test_demo_and_static_viewer_default_to_bundled_configuration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for options in (["--demo"], ["--depth-maps"], ["--ros", "--demo"]):
        args = parse_arguments(options)
        assert args.dataset is None and args.config is None and args.depth_maps
        assert args.replay == (DEMO_REPLAY if args.demo else None)
    profile = load_profile()
    assert Path(profile["configuration_source"]) == DEFAULT_CONFIG / "heightmaps.json"
    assert profile["ground_z_stage_m"] == -0.0245
    assert profile["front"]["cell_m"] == 0.0125
    assert profile["surround"]["cell_m"] == 0.01
    for name, rotation in (("cabin", 0), ("arm", 180)):
        sensor = profile["sensors"][name]
        assert sensor["rotation_deg"] == rotation
        assert sensor["depth_scale_m"] == pytest.approx(0.0001)
        assert (sensor["depth_intrinsics"]["width"], sensor["depth_intrinsics"]["height"]) == (640, 480)
    with pytest.raises(SystemExit):
        parse_arguments(["--config", "profile", "--dataset", "recording"])
    with pytest.raises(SystemExit):
        parse_arguments(["--demo", "--replay", "motion.json"])


def test_explicit_dataset_session_remains_supported_without_falling_back(tmp_path):
    (tmp_path / "meta").mkdir()
    session = tmp_path / "native/session-003"
    session.mkdir(parents=True)
    meta = json.loads((DEFAULT_CONFIG / "heightmaps.json").read_text())
    meta.pop("ground_z_stage_m")  # Recorded metadata has a native ground origin instead.
    meta["height_color_limit_m"] = 0.25
    (tmp_path / "meta/heightmaps.json").write_text(json.dumps(meta))
    for name, key in STREAMS.items():
        shutil.copyfile(DEFAULT_CONFIG / f"{name}.json", session / f"{key}.depth.json")
    profile = load_profile(tmp_path, 3)
    assert profile["height_color_limit_m"] == 0.25
    assert profile["ground_z_stage_m"] == -0.0245
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path, 4)
    with pytest.raises(ValueError, match="either"):
        load_profile(tmp_path, config=DEFAULT_CONFIG)


def test_demo_poses_move_the_scene_and_reset_without_editing_the_asset():
    from camera_calibration.usd_cameras import cameras, describe
    from scipy.spatial.transform import Rotation

    from pxr import Gf, Usd, UsdGeom

    scene_path = ROOT / "demo_env.usd"
    original = scene_path.read_bytes()
    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadNone)
    stage.Load("/World/excavator_bucket")
    stage.SetEditTarget(stage.GetSessionLayer())
    rig = cameras(stage)
    frames = load_replay(DEMO_REPLAY)
    assert DEMO_REPLAY.stat().st_size < 128 * 1024
    assert len(frames) == 110 and frames[0]["timestamp_ns"] == 0
    ops, positions = {}, []
    for frame in frames:
        for matrix in frame["transforms"].values():
            matrix = np.asarray(matrix)
            np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-6)
            np.testing.assert_array_equal(matrix[:, 3], [0, 0, 0, 1])
        apply_transforms(stage, frame["transforms"], ops)
        for camera in rig.values():
            optical = describe(camera)["optical_to_world"]
            world = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            np.testing.assert_allclose(optical["translation_m"], world.Transform(Gf.Vec3d(0)), atol=1e-12)
            assert np.linalg.norm(optical["quaternion_xyzw"]) == pytest.approx(1)
            np.testing.assert_allclose(
                Rotation.from_quat(optical["quaternion_xyzw"]).apply([0, 0, 1]),
                world.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized(),
                rtol=0,
                atol=2e-6,
            )
        positions.append(
            np.asarray(
                UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/excavator_bucket/tool_body"))
            )[3, :3].copy()
        )
    assert np.linalg.norm(np.ptp(positions, axis=0)) > 0.1
    apply_transforms(stage, frames[0]["transforms"], ops)
    reset = np.asarray(
        UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/excavator_bucket/tool_body"))
    )[3, :3]
    np.testing.assert_allclose(reset, positions[0])
    assert scene_path.read_bytes() == original


@pytest.mark.parametrize("bad_time", [-1, 0, 1.5])
def test_replay_rejects_invalid_or_repeated_timestamps(tmp_path, bad_time):
    frame = {"timestamp_ns": 0, "transforms": {"/robot": np.eye(4).tolist()}}
    plan = {"views": {"cabin": [frame, {**frame, "timestamp_ns": bad_time}]}}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="timestamps"):
        load_replay(path)


@pytest.mark.skipif(
    not (REPO / "native_stack/tools/heightmap_bench/mapper.py").is_file(), reason="Robot mapper required"
)
def test_simulator_uses_local_filter_overrides_and_tool_geometry(tmp_path):
    from robot_mapping import RobotPolicy, UsdRobotPose

    config = tmp_path / "profile"
    shutil.copytree(DEFAULT_CONFIG, config)
    meta = json.loads((config / "heightmaps.json").read_text())
    meta["arm_model"] = str(ROOT / "assets/arm_model.npz")
    meta["tool_tip_m"] = [0.01, 0.02, -0.1]
    (config / "heightmaps.json").write_text(json.dumps(meta))
    pipeline = json.loads((config / "pipeline.json").read_text())
    pipeline["projection"]["stride"] = 8
    (config / "pipeline.json").write_text(json.dumps(pipeline))
    policy = RobotPolicy(REPO, config=config)
    assert policy.projector.stride == 8
    assert policy.sources["pipeline"] == config / "pipeline.json"
    assert policy.config == config / "fastdem.yaml"
    assert policy.calibration["depth_correction"] == {}
    # A supplied simulator tip must not consult the external controller config.
    pose = UsdRobotPose(None, tmp_path / "missing-robot", -0.0245, tip=policy.tool_tip)
    np.testing.assert_array_equal(pose.tip, meta["tool_tip_m"])
