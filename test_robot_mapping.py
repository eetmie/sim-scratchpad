"""Integration checks against the actual robot policy and paused USD scene."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from robot_mapping import ROOT, STREAMS, RobotPolicy, UsdRobotPose

DATASET = Path(os.environ.get("KAIVURI_DATASET", Path.home() / "Desktop/heightmap_tests_complete"))
REPO = Path(os.environ.get("KAIVURI_REPO", Path.home() / "Documents/GitHub/kaivuriprokkis"))
REPLAY = ROOT / "logs/depth_replay/episode-002"
pytestmark = pytest.mark.skipif(
    not (REPLAY / "replay.json").is_file() or not REPO.is_dir(),
    reason="Local robot checkout and saved calibration replay required",
)


def recorded_state(policy, stamp):
    samples = {}
    with (DATASET / "native/episode-000002/samples.jsonl").open() as stream:
        for line in stream:
            state = json.loads(line)["status"]
            if state["fk_valid"] and state["imu_healthy"]:
                samples[state["imu_host_monotonic_ns"]] = state
    return policy.mapper.pose_at(stamp, np.array(sorted(samples)), [samples[t] for t in sorted(samples)], 0)


def test_live_usd_pivots_and_tip_match_recorded_fk_after_replay():
    from pxr import Usd

    from camera_calibration.recorded_poses import apply_transforms

    policy = RobotPolicy(REPO, DATASET)
    frame = json.loads((REPLAY / "replay.json").read_text())["views"]["cabin"][55]
    expected = recorded_state(policy, frame["timestamp_ns"])
    stage = Usd.Stage.Open(str(ROOT / "demo_env.usd"))
    stage.SetEditTarget(stage.GetSessionLayer())
    apply_transforms(stage, frame["transforms"], {})
    actual, _, _, _ = UsdRobotPose(stage, REPO, -0.0245).sample({})
    np.testing.assert_allclose(actual["fk_joint_xyz_m"], expected["fk_joint_xyz_m"], atol=1e-6)
    np.testing.assert_allclose(actual["fk_xyz_m"], expected["fk_xyz_m"], atol=0.001)
    np.testing.assert_allclose(actual["slew_yaw_rad"], expected["slew_yaw_rad"], atol=1e-6)


def test_filter_sweep_profiles_do_not_leak_their_global_arm_model():
    baseline = RobotPolicy(REPO, DATASET)
    smaller = RobotPolicy(REPO, DATASET, overrides={"arm_margins_m": [0.06, 0.035, 0.08]})
    frame = json.loads((REPLAY / "replay.json").read_text())["views"]["cabin"][55]
    state = recorded_state(baseline, frame["timestamp_ns"])
    poses, images = {}, {}
    for name in STREAMS:
        with np.load(REPLAY / name / f"frame-{frame['frame_index']:06d}.npz") as saved:
            poses[name], images[name] = saved["pose"].copy(), saved["z16"].copy()
        poses[name][:3, 3] -= [0, 0, -0.0245]
    for policy in (baseline, smaller):
        policy.set_optical_poses({name: state for name in STREAMS}, poses)
    first = baseline.scan("cabin", images["cabin"], state)
    other = smaller.scan("cabin", images["cabin"], state)
    again = baseline.scan("cabin", images["cabin"], state)
    np.testing.assert_array_equal(first[4], again[4])
    np.testing.assert_allclose(first[0], again[0], atol=1e-12)
    assert np.count_nonzero(other[4] == 2) <= np.count_nonzero(first[4] == 2)
