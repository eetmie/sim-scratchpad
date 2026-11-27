"""Fixed-rate hydraulic motion and USD joint kinematics for the sensor demo.

The external project's learned forward models are the free-space dynamics. USD
receives their integrated state; this does not simulate contact or hydraulic load.
Only the main thread touches USD or Kit input. The worker owns model history.
"""

import importlib.util
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
ARM_JOINTS = ("revolute_lift", "revolute_tilt", "revolute_tool")
SLEW = "revolute_carriage"
PITCH = "revolute_carriage_pitch"
JOINTS = (SLEW, *ARM_JOINTS, PITCH)
DT = 0.01
GAMEPAD_DEADZONE = 0.30  # Direct valve teleop in Isaac-hydraulic-actuator/sim.py.


def actuator_checkout(path=None):
    path = Path(path or os.environ.get("HYDRAULIC_ACTUATOR_REPO", ROOT.parent / "Isaac-hydraulic-actuator"))
    if not (path / "actuators/hydraulic_actuator.py").is_file():
        raise FileNotFoundError(f"Hydraulic actuator checkout not found: {path}; use --actuator-repo PATH")
    return path.resolve()


def load_source(path, name):
    """Import the external implementation without changing sys.path or copying it."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HydraulicMotion:
    """Authoritative state: slew, lift, tilt, tool, carriage pitch, in USD radians."""

    def __init__(self, initial, limits, *, dynamics=True, repo=None):
        self.initial = np.asarray(initial, dtype=np.float64).copy()
        self.limits = np.asarray(limits, dtype=np.float64)
        if self.initial.shape != (5,) or self.limits.shape != (5, 2):
            raise ValueError("Expected five motion channels and five joint limits")
        self.models, self.guard = [], None
        self.inference_ms = deque(maxlen=10000)
        self.steps = self.velocity_clips = self.limit_hits = 0
        self.peak_raw_velocity = 0.0
        if dynamics:
            repo = actuator_checkout(repo)
            import torch

            # Small single-sample networks are faster on one CPU thread; leave the GPU to RTX.
            torch.set_num_threads(1)
            net = load_source(repo / "actuators/hydraulic_actuator.py", "viewer_hydraulic_net").HydraulicActuatorNet
            for directory, indices, positions, commands in (
                (
                    "actuator",
                    [1, 2, 3, 4],
                    ["joint_pos_boom", "joint_pos_arm", "joint_pos_bucket", "carriage_pitch"],
                    ["combined_cmd_lift", "combined_cmd_tilt", "combined_cmd_scoop"],
                ),
                ("slew", [0], ["joint_pos_slew"], ["combined_cmd_rotate"]),
            ):
                path = repo / "models" / directory
                meta = json.loads((path / "model_meta.json").read_text())
                if (
                    meta["q_cols"] != positions
                    or meta["u_cols"] != commands
                    or meta["simulation_joint_names"] != [JOINTS[i] for i in indices]
                ):
                    raise ValueError(f"Unsupported actuator channel contract: {path}")
                self.models.append((net(path, device="cpu", sim_dt=DT), indices))
            guard = load_source(repo / "endstop_guard.py", "viewer_endstop_guard").EndStopGuard
            self.guard = guard(self.limits[1:4], DT)
        self.reset()

    def reset(self):
        self.position = self.initial.copy()
        self.velocity = np.zeros(5)
        for model, _ in self.models:
            model.reset()
        if self.guard is not None:
            self.guard.reset()

    def step(self, command):
        """Advance exactly 10 ms, independent of rendering and wall-clock delays."""
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (4,) or not np.isfinite(command).all() or np.any(np.abs(command) > 1):
            raise ValueError("Expected four finite valve demands in [-1, 1], slew/lift/tilt/tool")
        started = time.perf_counter()
        velocity = np.r_[command * [0.8, 0.5, 0.5, 0.5], 0.0]
        for model, indices in self.models:
            valves = command[1:4] if len(indices) == 4 else command[:1]
            velocity[indices] = model.predict_velocity(self.position[indices], self.velocity[indices], valves)
        self.inference_ms.append((time.perf_counter() - started) * 1000)
        if not np.isfinite(velocity).all():
            raise FloatingPointError(f"Non-finite hydraulic prediction at step {self.steps}: {velocity}")
        self.peak_raw_velocity = max(self.peak_raw_velocity, float(np.max(np.abs(velocity))))
        self.velocity_clips += int(np.any(np.abs(velocity) > 2.0))
        velocity = np.clip(velocity, -2.0, 2.0)  # same bound as the actuator project's demo
        if self.guard is not None:
            velocity[1:4] = self.guard.apply(self.position[1:4], velocity[1:4], command[1:4])
        proposed = self.position + DT * velocity
        self.position = np.clip(proposed, self.limits[:, 0], self.limits[:, 1])
        hit = proposed != self.position
        self.limit_hits += int(np.any(hit))
        self.velocity = np.where(hit, 0, velocity)
        self.steps += 1
        return self.position.copy()


class MotionWorker:
    """100 Hz worker with bounded catch-up; a render stall never changes model dt."""

    def __init__(self, motion):
        self.motion = motion
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.command = np.zeros(4)
        self.position = motion.position.copy()
        self.reset_pending = False
        self.error = None
        self.dropped_s = 0.0
        self.lateness_ms = deque(maxlen=10000)
        self.started = self.finished = None
        self.thread = threading.Thread(target=self._run, name="hydraulic-100hz", daemon=True)

    def start(self):
        self.started = time.perf_counter()
        self.thread.start()

    def _run(self):
        try:
            if self.motion.models:
                import torch

                # OpenMP's thread limit is local to the calling thread on Windows.
                torch.set_num_threads(1)
            due = time.perf_counter() + DT
            while not self.stop.is_set():
                if self.stop.wait(max(0, due - time.perf_counter())):
                    break
                now = time.perf_counter()
                late = max(0, now - due)
                self.lateness_ms.append(late * 1000)
                # Catch up ordinary frame jitter, but never replay seconds of old input.
                if late > 0.25:
                    self.dropped_s += late - 0.25
                    due = now - 0.25
                with self.lock:
                    command, reset = self.command.copy(), self.reset_pending
                    self.reset_pending = False
                if reset:
                    self.motion.reset()
                    due = now
                position = self.motion.step(command)
                with self.lock:
                    self.position = position
                due += DT
        except BaseException as error:
            self.error = error
        finally:
            self.finished = time.perf_counter()

    def sample(self, command, reset=False):
        if self.error is not None:
            raise RuntimeError("Hydraulic control worker failed") from self.error
        with self.lock:
            self.command = np.asarray(command, dtype=np.float32).copy()
            self.reset_pending |= reset
            return self.motion.initial.copy() if self.reset_pending else self.position.copy()

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("Hydraulic worker did not stop")

    def report(self):
        elapsed = (self.finished or time.perf_counter()) - self.started if self.started else 0
        return {
            "model_dt_s": DT,
            "model_steps": self.motion.steps,
            "wall_s": elapsed,
            "simulated_s": self.motion.steps * DT,
            "achieved_hz": self.motion.steps / elapsed if elapsed else 0,
            "dropped_wall_s": self.dropped_s,
            "inference_ms": percentiles(self.motion.inference_ms),
            "scheduling_lateness_ms": percentiles(self.lateness_ms),
            "velocity_clip_steps": self.motion.velocity_clips,
            "joint_limit_steps": self.motion.limit_hits,
            "peak_raw_velocity_rad_s": self.motion.peak_raw_velocity,
            "error": repr(self.error) if self.error else None,
        }


def percentiles(values):
    return dict(zip(("median", "p95", "max"), map(float, np.percentile(list(values), [50, 95, 100])))) if values else {}


class SceneJoints:
    """USD joint-frame FK, including the released model's virtual bearing pitch."""

    def __init__(self, stage):
        from camera_calibration.usd_cameras import ROBOT

        from pxr import UsdGeom, UsdPhysics

        self.stage, self.ops, self.joints = stage, {}, []
        self.root = ROBOT
        cache = UsdGeom.XformCache()
        self.root_inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT))).T)
        base_path = f"{ROBOT}/lower_carriage"
        self.base = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(base_path))).T
        self.initial = np.zeros(5)
        self.limits = np.array([[-np.inf, np.inf]] * 5)
        self.limits[4] = np.deg2rad([-3, 3])  # released actuator's rocking USD
        for name in (*JOINTS[:4], "fix_tool"):
            joint = UsdPhysics.Joint.Get(stage, f"{ROBOT}/Joints/{name}")
            if not joint:
                raise ValueError(f"Missing scene joint: {name}")
            parent, child = str(joint.GetBody0Rel().GetTargets()[0]), str(joint.GetBody1Rel().GetTargets()[0])
            frames = []
            for pos, quat in (
                (joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                (joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
            ):
                frame = np.eye(4)
                frame[:3, :3] = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
                frame[:3, 3] = pos
                frames.append(frame)
            axis, index = None, None
            if name in JOINTS:
                index = JOINTS.index(name)
                revolute = UsdPhysics.RevoluteJoint(joint)
                axis = np.eye(3)["XYZ".index(revolute.GetAxisAttr().Get())]
                p = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent))).T
                c = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child))).T
                relative = np.linalg.inv(p @ frames[0]) @ c @ frames[1]
                self.initial[index] = Rotation.from_matrix(relative[:3, :3]).as_rotvec() @ axis
                self.limits[index] = np.deg2rad(
                    [revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get()]
                )
            self.joints.append((parent, child, frames[0], np.linalg.inv(frames[1]), axis, index))

    def transforms(self, position):
        world = {f"{self.root}/lower_carriage": self.base}
        result = {}
        for parent, child, parent_frame, child_inverse, axis, index in self.joints:
            rotation = np.eye(4)
            if index is not None:
                rotation[:3, :3] = Rotation.from_rotvec(axis * position[index]).as_matrix()
                if index == 0:
                    # Rocking asset: bearing yaw -> roll (zero) -> pitch -> upper carriage.
                    rotation[:3, :3] @= Rotation.from_euler("y", position[4]).as_matrix()
            world[child] = world[parent] @ parent_frame @ rotation @ child_inverse
            result[child] = (self.root_inverse @ world[child]).T
        return result

    def apply(self, position):
        from scene_motion import apply_transforms

        apply_transforms(self.stage, self.transforms(position), self.ops)


class Gamepad:
    """Same valve signs/stick assignment as Isaac-hydraulic-actuator/sim.py."""

    def __init__(self):
        import omni.appwindow

        # Load the local input helper without shadowing the installed newton package.
        helper = load_source(ROOT / "newton/teleop.py", "viewer_teleop")
        self.window = omni.appwindow.get_default_app_window()
        self.source = helper.KitInput(self.window)
        self.source.description = self.source.description.replace("keyboard still works", "R resets the pose")
        self.deadzone = helper.apply_deadzone
        self.previous_reset = False

    def read(self):
        s = self.source
        s.gamepad = self.window.get_gamepad(0)  # allow connecting the controller after launch
        axes = [-s.stick("left_x"), s.stick("right_y"), -s.stick("left_y"), -s.stick("right_x")]
        command = np.array([self.deadzone(value, GAMEPAD_DEADZONE) for value in axes])
        down = s.button("b") or s.key("r")
        reset, self.previous_reset = down and not self.previous_reset, down
        return command, reset
