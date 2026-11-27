"""Compare the actual viewer recurrence, reference plant, recorded motion and stick response.

Run in the Isaac Lab Python environment. --native accepts a recording dataset with
native/episode-*/samples.jsonl for the optional empirical slew speed check.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydraulic_motion import DT, HydraulicMotion, MotionWorker, SceneJoints, actuator_checkout


def release_files(repo):
    hashes = {}
    for group in ("actuator", "slew"):
        directory = repo / "models" / group
        expected = json.loads((directory / "release_manifest.json").read_text())["files"]
        hashes[group] = {f: hashlib.sha256((directory / f).read_bytes()).hexdigest() for f in expected}
        if hashes[group] != expected:
            raise ValueError(f"Release fingerprint mismatch: {directory}")
    return hashes


def reference_rollout(motion, repo):
    from actuators import HydraulicActuatorNet
    from hydraulic_controller.core import DEFAULT_ASSET, HydraulicPlant
    from hydraulic_controller.kinematics import ExcavatorKinematics

    kin = ExcavatorKinematics(DEFAULT_ASSET, "cpu")
    np.testing.assert_allclose(kin.limits.numpy(), motion.limits[1:], atol=1e-6)
    arm = HydraulicPlant(repo / "models/actuator", kin, 1, "cpu")
    slew = HydraulicActuatorNet(repo / "models/slew", sim_dt=DT)
    profiles = {}
    for label, commands in (
        ("held_reversed_neutral", np.repeat([[1, 1, -1, 1], [-1, -1, 1, -1], [0, 0, 0, 0]], [350, 350, 200], axis=0)),
        ("mixed_20hz", np.repeat(np.random.default_rng(42).uniform(-1, 1, (100, 4)), 5, axis=0)),
    ):
        motion.reset()
        arm.reset(torch.tensor([0]), torch.tensor(motion.initial[None, 1:], dtype=torch.float32))
        slew.reset()
        sq = motion.initial[:1].astype(np.float32)
        sv = np.zeros_like(sq)
        errors = []
        for valve in commands.astype(np.float32):
            arm.u_cmd[0] = torch.from_numpy(valve[1:])
            arm.step()
            sv = np.clip(slew.predict_velocity(sq, sv, valve[:1]), -2, 2)
            sq = sq + DT * sv
            actual = motion.step(valve)
            errors.append(actual - np.r_[sq, arm.q[0].numpy()])
        profiles[label] = {
            "steps": len(commands),
            "max_abs_position_error_deg": np.rad2deg(np.abs(errors).max(axis=0)).tolist(),
        }
    return profiles


def pulse_speeds(motion):
    output = []
    for raw in (0.5, 0.75, 1.0):
        for zone in (0.20, 0.30):
            for direction in (-1, 1):
                speeds = []
                for axis in range(4):
                    motion.reset()
                    valve = np.zeros(4)
                    valve[axis] = direction * (raw - zone) / (1 - zone)
                    samples = []
                    for _ in range(60):
                        motion.step(valve)
                        samples.append(motion.velocity[axis])
                    speeds.append(float(np.rad2deg(np.mean(samples[-20:]))))
                output.append(dict(raw_stick=raw * direction, deadzone=zone, mean_deg_s_400_to_600ms=speeds))
    return output


def catchup_probe():
    """Pause the first worker tick, change the valve during the stall and timestamp its first applied step."""
    motion = HydraulicMotion(np.zeros(5), np.array([[-10, 10]] * 5), dynamics=False)
    entered, resume, applied = threading.Event(), threading.Event(), threading.Event()
    run_step = motion.step
    command_steps = []

    def delayed_step(command):
        if not entered.is_set():
            entered.set()
            if not resume.wait(2):
                raise TimeoutError("Probe did not resume")
        if command[0] and not applied.is_set():
            command_steps.append(motion.steps + 1)
            applied.set()
        return run_step(command)

    motion.step = delayed_step
    worker = MotionWorker(motion)
    worker.start()
    try:
        if not entered.wait(2):
            raise TimeoutError("Worker did not start")
        time.sleep(0.12)
        received = time.perf_counter() - worker.started
        worker.sample([1, 0, 0, 0])
        resume.set()
        if not applied.wait(2):
            raise TimeoutError("Worker did not apply command")
    finally:
        resume.set()
        worker.close()
    return dict(
        received_wall_s=received,
        first_applied_tick_s=command_steps[0] * DT,
        backdated_ms=1000 * (received - command_steps[0] * DT),
    )


def settled_slew(motion):
    speeds = {}
    for valve in (-1.0, -0.95, 0.95, 1.0):
        motion.reset()
        tail = []
        for step in range(200):
            motion.step([valve, 0, 0, 0])
            if step >= 60:
                tail.append(motion.velocity[0])
        speeds[str(valve)] = float(np.rad2deg(np.mean(tail)))
    return speeds


def recording_trials(motion, repo):
    root = repo / "dataset/complete_dataset"
    manifest = json.loads((root / "dataset.json").read_text())
    trials = []
    for recording in manifest["recordings"]:
        # Two predetermined offsets in each recording's longest contiguous chunk.
        # Pick by duration, independent of model error or observed speed.
        def rows(name):
            with np.load(root / name) as data:
                return len(data["t"])

        name = max(recording["training_paths"], key=rows)
        with np.load(root / name) as data:
            q, v, u, rotate, t = (data[k] for k in ("q", "v", "u", "u_rotate", "t"))
        np.testing.assert_allclose(np.diff(t), DT, atol=1e-8, rtol=0)
        for fraction in (0.25, 0.75):
            duration = 200
            start = 61 + int((len(q) - 62 - duration) * fraction)
            if start < 61:
                raise ValueError(f"Chunk too short: {name}")
            motion.reset()
            # Prime velocity and command histories with measured samples through start-1.
            model = motion.models[0][0]
            for i in range(start - 61, start):
                model.predict_velocity(q[i], v[i], u[i])
            motion.position[1:] = q[start]
            motion.velocity[1:] = v[start]
            predicted = [motion.position[1:].copy()]
            for i in range(start, start + duration):
                predicted.append(motion.step(np.r_[rotate[i], u[i]])[1:])
            truth = q[start : start + duration + 1]
            predicted = np.asarray(predicted)
            # Compare traveled angles on the same 250 ms grid as the GUI. This suppresses gyro noise.
            observed = np.rad2deg(np.diff(truth[::25], axis=0))
            modeled = np.rad2deg(np.diff(predicted[::25], axis=0))
            trials.append(
                dict(
                    recording=recording["id"],
                    chunk=name,
                    start=start,
                    observed_deg=observed.tolist(),
                    modeled_deg=modeled.tolist(),
                    mae_deg=np.rad2deg(np.abs(predicted[1:] - truth[1:]).mean(axis=0)).tolist(),
                )
            )
    measured = np.concatenate([row["observed_deg"] for row in trials])
    predicted = np.concatenate([row["modeled_deg"] for row in trials])
    axes = {}
    for i, axis in enumerate(("lift", "tilt", "bucket", "carriage_pitch")):
        m, p = measured[:, i], predicted[:, i]
        axes[axis] = dict(
            observed_mean_abs_deg_s=float(np.abs(m).mean() * 4),
            modeled_mean_abs_deg_s=float(np.abs(p).mean() * 4),
            ratio_of_total_travel=float(np.abs(p).sum() / np.abs(m).sum()),
            path_mae_deg=float(np.mean([r["mae_deg"][i] for r in trials])),
        )
    return {"window_seconds": 2, "recordings": len(manifest["recordings"]), "axes": axes, "windows": trials}


def slew_observations(root):
    """Measure yaw speed over non-overlapping 250 ms intervals after 600 ms at >=95% command."""
    by_sign = {"negative": [], "positive": []}
    for file in sorted(root.glob("native/episode-*/samples.jsonl")):
        sign = 0
        begin = None
        sample_from = 0
        previous = 0
        with file.open() as stream:
            for line in stream:
                s = json.loads(line)["status"]
                u = s["applied_action"][0]
                stamp = s["imu_host_monotonic_ns"] * 1e-9
                valid = (
                    s["imu_healthy"]
                    and s["enable_slew"]
                    and s["live_pwm"]
                    and s["pump_on"]
                    and s["state"] == "armed"
                    and not s["simulated_imu"]
                    and abs(u) >= 0.95
                )
                if not valid or not 0 < stamp - previous <= 0.05 or np.sign(u) != sign:
                    begin = None
                    sample_from = stamp + 0.6
                    sign = np.sign(u) if valid else 0
                previous = stamp
                if valid:
                    if stamp >= sample_from:
                        if begin is None:
                            begin = (stamp, s["slew_yaw_rad"])
                        elif stamp - begin[0] >= 0.25:
                            angle = (s["slew_yaw_rad"] - begin[1] + np.pi) % (2 * np.pi) - np.pi
                            by_sign["positive" if sign > 0 else "negative"].append(
                                float(np.rad2deg(angle) / (stamp - begin[0]))
                            )
                            begin = (stamp, s["slew_yaw_rad"])
    return {
        k: {"intervals": len(v), "yaw_deg_s_p10_median_p90": np.percentile(v, [10, 50, 90]).tolist() if v else []}
        for k, v in by_sign.items()
    }


def main():
    from pxr import Usd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actuator-repo", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "logs/gamepad/audit/results.json")
    args = parser.parse_args()
    repo = actuator_checkout(args.actuator_repo)
    sys.path.insert(0, str(repo))
    stage = Usd.Stage.Open(str(ROOT / "demo_env.usd"), load=Usd.Stage.LoadNone)
    stage.Load("/World/excavator_bucket")
    rig = SceneJoints(stage)
    motion = HydraulicMotion(rig.initial, rig.limits, repo=repo)
    result = dict(
        channel_order=["slew", "lift", "tilt", "bucket", "carriage_pitch"],
        release_sha256=release_files(repo),
        initial_q_rad=rig.initial.tolist(),
        parity=reference_rollout(motion, repo),
        stick_response=pulse_speeds(motion),
        robot=recording_trials(motion, repo),
        simulated_slew_deg_s=settled_slew(motion),
        catchup=catchup_probe(),
    )
    if args.native:
        result["recorded_slew"] = slew_observations(args.native)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
