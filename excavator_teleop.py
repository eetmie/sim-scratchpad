"""Gamepad teleop for the excavator + today's cabin lidar height map.

Run in Isaac Sim's Script Editor with demo_env.usd open, then press Play. Rerun to restart both.
Sticks are rate commands: each joint target moves at stick * speed and holds when released.

ISO excavator layout:
    left stick  X: swing (carriage)      left stick  Y: arm (tilt boom)
    right stick X: bucket (tool)         right stick Y: boom (lift)
    Y button: clear the lidar height map
"""

import asyncio
import time

import carb
import carb.input
import omni.appwindow
import omni.kit.app
import omni.timeline
import omni.usd

# The lidar helper (object-ID self-filter, remembered map, stale-cell clearing) stays in one file.
LIDAR_SCRIPT = "/home/joel/IsaacLab/scripts/sim-scratchpad/lidar_demo_script.py"
JOINTS_PATH = "/World/excavator_bucket/Joints"
GAMEPAD_CAMERA_SETTING = "/persistent/app/omniverse/gamepadCameraControl"
DEADZONE = 0.20
MAX_STEP_S = 0.1  # Caps dt so a frame hitch can't jump the arm.

_P = carb.input.GamepadInput
# joint: (positive input, negative input, speed in deg/s). Flip a speed's sign if a joint moves backwards.
JOINT_AXES = {
    "revolute_carriage": (_P.LEFT_STICK_RIGHT, _P.LEFT_STICK_LEFT, -15.0),
    "revolute_tilt": (_P.LEFT_STICK_UP, _P.LEFT_STICK_DOWN, -12.5),
    "revolute_tool": (_P.RIGHT_STICK_RIGHT, _P.RIGHT_STICK_LEFT, -20.0),
    "revolute_lift": (_P.RIGHT_STICK_UP, _P.RIGHT_STICK_DOWN, 10.0),
}


def apply_deadzone(value):
    """Zero small stick noise, then rescale so the output still spans -1..1."""
    magnitude = (abs(value) - DEADZONE) / (1.0 - DEADZONE)
    return 0.0 if magnitude <= 0.0 else min(magnitude, 1.0) * (1.0 if value > 0 else -1.0)


def integrate_target(target, command, speed, dt, lower, upper):
    """Rate control: move the drive target by command * speed * dt, kept inside the joint limits."""
    return min(max(target + command * speed * dt, lower), upper)


def stop_excavator_teleop():
    """Stop the joystick loop and give the gamepad back to the viewport camera."""
    state = globals().pop("excavator_teleop_state", None)
    if state is not None:
        state["task"].cancel()
        carb.settings.get_settings().set(GAMEPAD_CAMERA_SETTING, state["camera_setting"])


async def drive_joints(state):
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    gamepad_input = carb.input.acquire_input_interface()
    gamepad = omni.appwindow.get_default_app_window().get_gamepad(0)
    joints = state["joints"]
    targets = {}
    was_playing = False
    clear_was_down = False
    last = time.monotonic()
    while True:
        await app.next_update_async()
        now = time.monotonic()
        dt, last = min(now - last, MAX_STEP_S), now
        if not timeline.is_playing():
            was_playing = False
            continue
        if not was_playing:
            # Start from whatever the drives hold now (Stop may have reset them).
            targets = {name: float(joint["target"].Get()) for name, joint in joints.items()}
            was_playing = True

        for name, (positive, negative, speed) in JOINT_AXES.items():
            raw = gamepad_input.get_gamepad_value(gamepad, positive) - gamepad_input.get_gamepad_value(gamepad, negative)
            command = apply_deadzone(raw)
            if command:
                joint = joints[name]
                targets[name] = integrate_target(targets[name], command, speed, dt, joint["lower"], joint["upper"])
                joint["target"].Set(targets[name])

        clear_down = gamepad_input.get_gamepad_value(gamepad, _P.Y) > 0.5
        lidar = globals().get("cabin_lidar_state")
        if clear_down and not clear_was_down and lidar is not None:
            lidar["heightmap"] = new_heightmap_memory()  # noqa: F821 - defined by the lidar script
            lidar["view"].update(lidar["heightmap"])
            print("[Teleop] Height map cleared.")
        clear_was_down = clear_down


def start_excavator_teleop():
    stage = omni.usd.get_context().get_stage()
    joints = {}
    for name in JOINT_AXES:
        prim = stage.GetPrimAtPath(f"{JOINTS_PATH}/{name}") if stage else None
        if not prim:
            raise RuntimeError(f"Open demo_env.usd first; missing joint {JOINTS_PATH}/{name}.")
        target = prim.GetAttribute("drive:angular:physics:targetPosition")
        if not target:
            raise RuntimeError(f"{name} has no angular position drive to command.")
        joints[name] = {
            "target": target,  # degrees, like the limits
            "lower": prim.GetAttribute("physics:lowerLimit").Get(),
            "upper": prim.GetAttribute("physics:upperLimit").Get(),
        }

    settings = carb.settings.get_settings()
    state = {"joints": joints, "camera_setting": settings.get(GAMEPAD_CAMERA_SETTING)}
    # Otherwise the same sticks also fly the viewport camera.
    settings.set(GAMEPAD_CAMERA_SETTING, False)
    state["task"] = asyncio.ensure_future(drive_joints(state))

    gamepad = omni.appwindow.get_default_app_window().get_gamepad(0)
    name = carb.input.acquire_input_interface().get_gamepad_name(gamepad) if gamepad else ""
    print(f"[Teleop] Gamepad 0: {name or 'NOT FOUND - connect it, then rerun'}. Press Play to drive.")
    return state


# Lidar first (it restarts itself on rerun), sharing this namespace so cabin_lidar_state is visible here.
exec(compile(open(LIDAR_SCRIPT).read(), LIDAR_SCRIPT, "exec"), globals())
stop_excavator_teleop()
excavator_teleop_state = start_excavator_teleop()
