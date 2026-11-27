"""Control timing, joint-frame geometry and compact camera calibration."""

import time
from pathlib import Path

import numpy as np
import pytest
from hydraulic_motion import DT, HydraulicMotion, MotionWorker, SceneJoints
from ir_views import scaled_camera
from launch_3dgs import parse_arguments

ROOT = Path(__file__).resolve().parents[1]


def test_gamepad_cli_and_replay_are_separate():
    args = parse_arguments(["--mode", "gamepad"])
    assert args.depth_maps and not args.nodynamics and not args.show_viewport
    assert args.replay is None and args.ir_width == 224
    assert parse_arguments(["--mode", "gamepad", "--nodynamics"]).nodynamics
    assert parse_arguments(["--replay", "custom.json"]).mode == "replay"
    live = parse_arguments(["--mode", "gamepad", "--ros", "--ros-cameras"])
    assert (live.ros_time, live.ir_width) == ("wall", 224)
    for argv in (["--nodynamics"], ["--ir-width", "0"]):
        with pytest.raises(SystemExit):
            parse_arguments(argv)


def test_preview_preserves_projected_pixel_centres_and_original_calibration():
    info = {
        "intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 390.0,
            "fy": 385.0,
            "cx": 323.0,
            "cy": 237.0,
            "coeffs": [0.1, 0, 0, 0, 0],
        }
    }
    intr = scaled_camera(info, 224)["intrinsics"]
    assert (intr["width"], intr["height"]) == (224, 168)
    point = np.array([0.1, -0.2]) / 0.8
    full = point * [390, 385] + [323, 237]
    small = point * [intr["fx"], intr["fy"]] + [intr["cx"], intr["cy"]]
    np.testing.assert_allclose(small, (full + 0.5) * 0.35 - 0.5)
    assert info["intrinsics"]["width"] == 640


def scene():
    from pxr import Usd

    stage = Usd.Stage.Open(str(ROOT / "demo_env.usd"), load=Usd.Stage.LoadNone)
    stage.Load("/World/excavator_bucket")
    stage.SetEditTarget(stage.GetSessionLayer())
    return stage


def test_joint_fk_reconstructs_initial_cad_pose_and_keeps_pins_connected():
    from pxr import UsdGeom

    stage = scene()
    rig = SceneJoints(stage)
    initial = rig.transforms(rig.initial)
    for path, matrix in initial.items():
        expected = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()
        np.testing.assert_allclose(matrix, expected, atol=7e-7)
    q = rig.initial + [0.3, 0.1, -0.2, 0.2, 0.02]
    rig.apply(q)
    cache = UsdGeom.XformCache()
    for parent, child, pframe, cinverse, _, _ in rig.joints:
        p = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent))).T
        c = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child))).T
        np.testing.assert_allclose((p @ pframe)[:3, 3], (c @ np.linalg.inv(cinverse))[:3, 3], atol=1e-8)
    rig.apply(rig.initial)
    for path, matrix in initial.items():
        np.testing.assert_allclose(UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation(), matrix)


def test_limit_contact_has_zero_velocity_and_reset_clears_state():
    motion = HydraulicMotion(np.zeros(5), np.array([[-0.02, 0.02]] * 5), dynamics=False)
    for _ in range(10):
        motion.step(np.ones(4))
    np.testing.assert_allclose(motion.position, [0.02] * 4 + [0])
    np.testing.assert_array_equal(motion.velocity, 0)
    motion.step(-np.ones(4))
    assert np.all(motion.position[:4] < 0.02)
    motion.reset()
    np.testing.assert_array_equal(motion.position, 0)
    with pytest.raises(ValueError, match="finite"):
        motion.step([float("nan")] * 4)


def test_worker_advances_while_main_thread_waits_for_a_slow_frame():
    motion = HydraulicMotion(np.zeros(5), np.array([[-10, 10]] * 5), dynamics=False)
    worker = MotionWorker(motion)
    worker.sample([1, 0, 0, 0])
    worker.start()
    try:
        time.sleep(0.35)  # no render/input callbacks during this interval
        state = worker.sample([1, 0, 0, 0])
    finally:
        worker.close()
    assert state[0] > 0.08  # at least ten updates, not one update per render
    assert worker.error is None
    assert motion.position[0] == pytest.approx(motion.steps * DT * 0.8)


def test_worker_reports_model_failure():
    motion = HydraulicMotion(np.zeros(5), np.array([[-10, 10]] * 5), dynamics=False)
    worker = MotionWorker(motion)
    worker.sample([float("nan")] * 4)
    worker.start()
    worker.thread.join(timeout=2)
    worker.close()
    with pytest.raises(RuntimeError, match="worker failed"):
        worker.sample(np.zeros(4))
    assert worker.report()["error"]
