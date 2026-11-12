"""Excavator sim <-> ROS 2. Run in Isaac Sim's Script Editor (or via open_demo.py), then press Play.

Start Isaac Sim with ros2/run_isaacsim.sh (sets up the bundled Jazzy ROS 2 libs), open demo_env.usd, then run this
file. Rerun to restart. The sim plays the robot: it drives the joints from /excavator/command and publishes what
the robot's sensors would. Everything else (teleop, self-filter, height map, joint_angles) is ROS nodes, the same
as on the robot. Topics, frames and camera mounts are in sensors.py.

Subscribed:
    /excavator/command                                std_msgs/Float32MultiArray [slew, lift, tilt, scoop] in
                                                      [-1, 1], rate demand; older than 150 ms = stop
Published (stamped with sim time):
    /clock                                            rosgraph_msgs/Clock
    /joint_states                                     sensor_msgs/JointState (rad), feeds robot_state_publisher
    /excavator/cameras/{cabin,arm}/image_raw          sensor_msgs/Image rgb8, CAMERA_HZ
    /excavator/cameras/{cabin,arm}/depth/image_raw    sensor_msgs/Image 16UC1 mm, stereo-like noise, 0 = no depth
    /excavator/cameras/{cabin,arm}/depth/camera_info  frame {cabin,arm}_camera (optical)
    /lidar_cabin/points, /lidar_boom/points           only with EXCAVATOR_LIDARS=1: PointCloud2 in the lidar's
                                                      frame, excavator hits included
EXCAVATOR_PROFILE=1 prints the app frame rate and the ms per bridge step every 10 s.
"""

import asyncio
import importlib
import math
import os
import sys
import time

import numpy as np

import carb
import carb.settings
import omni.graph.core as og
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

sys.path.insert(0, "/home/joel/IsaacLab/scripts/sim-scratchpad/ros2")  # no __file__ when exec()d from the Script Editor
import sensors  # noqa: E402 - shared with the ROS-side launch files

sensors = importlib.reload(sensors)  # a rerun in the Script Editor picks up edits (e.g. camera mounts)

ROBOT_PATH = "/World/excavator_bucket"
JOINTS_PATH = f"{ROBOT_PATH}/Joints"
GRAPH_PATH = "/ROS2Bridge"
MAX_STEP_S = 0.1  # caps dt so a frame hitch can't jump the arm
GAMEPAD_CAMERA_SETTING = "/persistent/app/omniverse/gamepadCameraControl"
LIDARS = os.environ.get("EXCAVATOR_LIDARS") == "1"
PROFILE = os.environ.get("EXCAVATOR_PROFILE") == "1"
# Passive stereo (emitter off) noise: sigma_z = z^2 * subpixel / (fx * baseline). D435 baseline 50 mm.
STEREO_BASELINE_M = 0.05
STEREO_SUBPIXEL = 0.1  # textured sand without the projector; tune against a real recording
DEPTH_MIN_M = 0.17  # D435 min-Z at 640 x 480
DEPTH_DROPOUT = 0.02  # random holes
EDGE_JUMP_M = 0.03  # 3x3 depth jumps above this are object edges: flying pixels or holes there
DEPTH_PIXEL_MM = 0.005  # USD camera sensor pitch; only the ratio focal length / aperture matters
JOINT_STATE_FIELDS = (
    "jointNames",
    "jointPositions",
    "jointVelocities",
    "jointEfforts",
    "jointDofTypes",
    "stageMetersPerUnit",
    "sensorTime",
)

# Without run_isaacsim.sh the bridge silently falls back to bundled Humble, which Lyrical may not talk to.
if os.environ.get("ROS_DISTRO") != "jazzy":
    raise RuntimeError(f"ROS_DISTRO={os.environ.get('ROS_DISTRO')!r}; restart Isaac Sim with ros2/run_isaacsim.sh.")
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

import rclpy  # noqa: E402 - bundled Jazzy rclpy, on sys.path once the bridge is enabled
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Float32MultiArray, Header  # noqa: E402

from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

if LIDARS:
    import sim_lidar  # noqa: E402 - enables the RTX lidar extension, so only when asked

    sim_lidar = importlib.reload(sim_lidar)

XYZ_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]


def build_graph(stage):
    """Action graph for /clock and /joint_states. Rebuilt on rerun; lives on the session layer (never saved)."""
    keys = og.Controller.Keys
    nodes = [
        ("Tick", "omni.graph.action.OnPlaybackTick"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("ReadJoints", "isaacsim.sensors.physics.IsaacReadJointState"),
        ("PublishJoints", "isaacsim.ros2.bridge.ROS2PublishJointState"),
    ]
    values = [
        ("SimTime.inputs:resetOnStop", True),
        ("ReadJoints.inputs:prim", [Sdf.Path(ROBOT_PATH)]),
        ("PublishJoints.inputs:topicName", "joint_states"),
    ]
    connects = [
        ("Tick.outputs:tick", "Clock.inputs:execIn"),
        ("Context.outputs:context", "Clock.inputs:context"),
        ("SimTime.outputs:simulationTime", "Clock.inputs:timeStamp"),
        ("Tick.outputs:tick", "ReadJoints.inputs:execIn"),
        ("Context.outputs:context", "PublishJoints.inputs:context"),
        ("SimTime.outputs:simulationTime", "PublishJoints.inputs:timeStamp"),
        ("ReadJoints.outputs:execOut", "PublishJoints.inputs:execIn"),
    ]
    connects += [(f"ReadJoints.outputs:{f}", f"PublishJoints.inputs:{f}") for f in JOINT_STATE_FIELDS]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        if stage.GetPrimAtPath(GRAPH_PATH):
            stage.RemovePrim(GRAPH_PATH)
        og.Controller.edit(
            {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
            {keys.CREATE_NODES: nodes, keys.SET_VALUES: values, keys.CONNECT: connects},
        )


def apply_camera(camera, xyz, q, fx):
    """D435i pose and lens on a USD camera: xyz + optical-frame quaternion (x, y, z, w) from sensors.py."""
    x, y, z, w = q
    width, height = sensors.DEPTH_SIZE
    camera.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*xyz))
    # USD cameras look down -Z with +Y up: optical frame turned 180 deg about X (q * (1, 0, 0, 0)).
    camera.GetPrim().GetAttribute("xformOp:orient").Set(Gf.Quatd(-x, Gf.Vec3d(w, z, -y)))
    camera.GetFocalLengthAttr().Set(fx * DEPTH_PIXEL_MM)
    camera.GetHorizontalApertureAttr().Set(width * DEPTH_PIXEL_MM)
    camera.GetVerticalApertureAttr().Set(height * DEPTH_PIXEL_MM)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))


def create_cameras(stage):
    """Render the D435i cameras of demo_env.usd (sensors.CAMERA_PRIMS), posed and lensed from sensors.py."""
    import omni.replicator.core as rep

    cameras = []
    width, height = sensors.DEPTH_SIZE
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, parent, xyz, q, fx, _ in sensors.DEPTH_CAMERAS:
            path = f"{ROBOT_PATH}/{parent}/{sensors.CAMERA_PRIMS[name]}"
            camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
            if not camera:
                raise RuntimeError(f"{path} must be a Camera in demo_env.usd")
            apply_camera(camera, xyz, q, fx)
            product = rep.create.render_product(path, (width, height), force_new=True, name=f"Camera_{name}")
            depth = rep.annotators.get("distance_to_image_plane", device="cpu")
            color = rep.annotators.get("LdrColor", device="cpu")
            depth.attach([product.path])
            color.attach([product.path])
            cameras.append({"name": name, "fx": fx, "product": product, "depth": depth, "color": color})
    return cameras


def stereo_depth_mm(z, fx, rng):
    """Clean z depth (m) -> D435i-like Z16 (mm): range-squared noise, holes, edge artifacts, min-Z."""
    z = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
    padded = np.pad(z, 1, mode="edge")
    shifts = [
        padded[1 + dy : 1 + dy + z.shape[0], 1 + dx : 1 + dx + z.shape[1]] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    ]
    edge = np.max(shifts, axis=0) - np.min(shifts, axis=0) > EDGE_JUMP_M
    # Flying pixels: edge depths land between foreground and background; half of the edge has no depth at all.
    z = np.where(edge, np.mean(shifts, axis=0), z)
    z = z + rng.standard_normal(z.shape, dtype=np.float32) * z * z * STEREO_SUBPIXEL / (fx * STEREO_BASELINE_M)
    hole = (rng.random(z.shape) < DEPTH_DROPOUT) | (edge & (rng.random(z.shape) < 0.5)) | (z < DEPTH_MIN_M)
    return np.where(hole, 0, np.clip(np.rint(z * 1000.0), 0, 65535)).astype(np.uint16)


def publish_images(state, header):
    """The rendered RGB image, at CAMERA_HZ."""
    width, height = sensors.DEPTH_SIZE
    for cam in state["cameras"]:
        rgba = cam["color"].get_data()
        if rgba is None or not np.size(rgba):
            continue
        header.frame_id = sensors.camera_frame(cam["name"])
        rgb = np.ascontiguousarray(np.asarray(rgba).reshape(height, width, 4)[..., :3])
        image = Image(header=header, height=height, width=width, encoding="rgb8", step=3 * width, data=rgb.tobytes())
        state["pubs"][cam["name"]]["image"].publish(image)


def publish_depth(state, header):
    """Noisy Z16 depth + camera_info, at DEPTH_HZ."""
    width, height = sensors.DEPTH_SIZE
    for cam in state["cameras"]:
        z = cam["depth"].get_data()
        if z is None or not np.size(z):
            continue
        name, fx = cam["name"], cam["fx"]
        header.frame_id = sensors.camera_frame(name)
        mm = stereo_depth_mm(np.asarray(z).reshape(height, width), fx, state["rng"])
        pubs = state["pubs"][name]
        pubs["depth"].publish(
            Image(header=header, height=height, width=width, encoding="16UC1", step=2 * width, data=mm.tobytes())
        )
        # shortcut: principal point at the image centre (the real ones are ~5 px off); sim pixels match this info.
        cx, cy = width / 2 - 0.5, height / 2 - 0.5
        pubs["info"].publish(
            CameraInfo(
                header=header,
                height=height,
                width=width,
                distortion_model="plumb_bob",
                d=[0.0] * 5,
                k=[fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0],
                r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                p=[fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
        )


def publish_lidars(state, header):
    """Each lidar's newest scan, in its own frame (excavator hits included, like a real lidar)."""
    for lidar in state["lidars"]:
        points = sim_lidar.new_scan(lidar)
        if points is not None:
            points = np.ascontiguousarray(points, dtype=np.float32)
            cloud = PointCloud2(
                header=Header(stamp=header.stamp, frame_id=lidar["name"]),
                height=1,
                width=len(points),
                fields=XYZ_FIELDS,
                point_step=12,
                row_step=12 * len(points),
                is_dense=True,
                data=points.tobytes(),
            )
            state["pubs"][lidar["name"]]["points"].publish(cloud)


def joint_drives(stage):
    """(target attribute, lower, upper) per command axis, in degrees like the USD drives.

    Limits are the USD joint limits narrowed to the robot's joint_angles bounds (kept 1 deg inside, so tracking
    overshoot stays in range): the sim can't reach poses the real machine can't, and joint_angles stays valid.
    Slew has no bound (the real machine turns freely; joint_angles wraps it to [-pi, pi]).
    """
    drives = []
    for i, name in enumerate(sensors.COMMAND_JOINTS):
        prim = stage.GetPrimAtPath(f"{JOINTS_PATH}/{name}")
        target = prim.GetAttribute("drive:angular:physics:targetPosition") if prim else None
        if not target:
            raise RuntimeError(f"Open demo_env.usd first; no position drive on {JOINTS_PATH}/{name}.")
        # native = sign * (URDF - offset) within [min, max]  ->  URDF range
        ends = [
            b / sensors.NATIVE_ANGLE_SIGN[i] + sensors.URDF_OFFSET[i]
            for b in (sensors.JOINT_ANGLE_MIN[i], sensors.JOINT_ANGLE_MAX[i])
        ]
        lower, upper = prim.GetAttribute("physics:lowerLimit").Get(), prim.GetAttribute("physics:upperLimit").Get()
        if max(ends) - min(ends) < 2 * math.pi - 1e-6:  # slew spans the full circle: it just keeps turning
            lower, upper = max(lower, math.degrees(min(ends)) + 1.0), min(upper, math.degrees(max(ends)) - 1.0)
        drives.append((target, lower, upper))
    return drives


def on_command(state, msg):
    """Keep the latest valid command; anything else is ignored, so the 150 ms timeout stops the arm."""
    if len(msg.data) == len(sensors.COMMAND_JOINTS) and all(math.isfinite(v) for v in msg.data):
        state["command"] = (time.monotonic(), np.clip(np.asarray(msg.data, dtype=float), -1.0, 1.0))


async def drive_joints(state):
    """Rate control like the valves: each target moves at command * speed and holds at zero or stale command."""
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    targets, last = None, time.monotonic()
    speed = np.asarray(sensors.COMMAND_SIGN) * np.asarray(sensors.COMMAND_SPEED_DEG_S)
    while True:
        await app.next_update_async()
        rclpy.spin_once(state["node"], timeout_sec=0)  # delivers /excavator/command
        now = time.monotonic()
        dt, last = min(now - last, MAX_STEP_S), now
        if not timeline.is_playing():
            targets = None
            continue
        if targets is None:  # start from what the drives hold now (Stop may have reset them)
            targets = [float(target.Get()) for target, _, _ in state["drives"]]
        stamp, command = state["command"]
        if now - stamp > sensors.COMMAND_TIMEOUT_S or not command.any():
            continue
        for i, (target, lower, upper) in enumerate(state["drives"]):
            if command[i]:
                targets[i] = min(max(targets[i] + command[i] * speed[i] * dt, lower), upper)
                target.Set(targets[i])


def timed(state, key, fn, *args):
    start = time.perf_counter()
    fn(*args)
    total, count = state["timing"].get(key, (0.0, 0))
    state["timing"][key] = (total + time.perf_counter() - start, count + 1)


def report_timing(state):
    """With EXCAVATOR_PROFILE=1: every 10 s, app frame rate and mean ms per bridge step."""
    state["frames"] += 1
    now = time.monotonic()
    if not PROFILE or now - state["timing_since"] < 10.0:
        return
    parts = ", ".join(f"{k} {1000 * t / n:.1f} ms x{n}" for k, (t, n) in sorted(state["timing"].items()))
    print(f"[ROS2 bridge] {state['frames'] / (now - state['timing_since']):.1f} app fps | {parts}", flush=True)
    state["timing"], state["timing_since"], state["frames"] = {}, now, 0


async def forward_sensors(state):
    """Images at CAMERA_HZ and depth at DEPTH_HZ of sim time; lidar scans as they come."""
    app = omni.kit.app.get_app()
    last_error = None
    while True:
        await app.next_update_async()
        sim_time = SimulationManager.get_simulation_time()
        header = Header()
        header.stamp.sec, header.stamp.nanosec = divmod(round(sim_time * 1e9), 1_000_000_000)  # round, like /clock
        try:
            # abs(): sim time restarts at 0 after Stop. 0.99: a 1/60 s step lands just short of 1/30 s.
            if abs(sim_time - state["last_image"]) >= 0.99 / sensors.CAMERA_HZ:
                state["last_image"] = sim_time
                timed(state, "image", publish_images, state, Header(stamp=header.stamp))
            if abs(sim_time - state["last_depth"]) >= 0.99 / sensors.DEPTH_HZ:
                state["last_depth"] = sim_time
                timed(state, "depth", publish_depth, state, Header(stamp=header.stamp))
            if state["lidars"]:
                timed(state, "lidar", publish_lidars, state, header)
            report_timing(state)
        except Exception as exc:  # keep publishing; an unhandled error would end this task without a word
            if repr(exc) != last_error:
                carb.log_error(f"[ROS2 bridge] sensor publish failed: {exc!r}")
                last_error = repr(exc)


def stop_ros2_bridge():
    state = globals().pop("ros2_bridge_state", None)
    if state is not None:
        state["task"].cancel()
        state["drive_task"].cancel()
        carb.settings.get_settings().set(GAMEPAD_CAMERA_SETTING, state["camera_setting"])
        if state["lidars"]:
            sim_lidar.destroy(state["lidars"])
        for cam in state["cameras"]:
            cam["depth"].detach([cam["product"].path])
            cam["color"].detach([cam["product"].path])
            cam["product"].destroy()
        state["node"].destroy_node()


def start_ros2_bridge():
    stage = omni.usd.get_context().get_stage()
    build_graph(stage)
    if not rclpy.ok():
        rclpy.init()
    node = Node("excavator_sim_bridge")
    settings = carb.settings.get_settings()
    state = {
        "node": node,
        "drives": joint_drives(stage),
        "command": (0.0, np.zeros(len(sensors.COMMAND_JOINTS))),
        "camera_setting": settings.get(GAMEPAD_CAMERA_SETTING),
        "cameras": create_cameras(stage),
        "lidars": sim_lidar.create(stage, ROBOT_PATH) if LIDARS else [],
        "pubs": {},
        "last_depth": -1.0,
        "last_image": -1.0,
        "rng": np.random.default_rng(0),
        "timing": {},
        "timing_since": time.monotonic(),
        "frames": 0,
    }
    for name, *_ in sensors.DEPTH_CAMERAS:
        state["pubs"][name] = {
            "image": node.create_publisher(Image, sensors.camera_topic(name, "image_raw"), 5),
            "depth": node.create_publisher(Image, sensors.camera_topic(name, "depth/image_raw"), 5),
            "info": node.create_publisher(CameraInfo, sensors.camera_topic(name, "depth/camera_info"), 5),
        }
    for lidar in state["lidars"]:
        state["pubs"][lidar["name"]] = {"points": node.create_publisher(PointCloud2, f"{lidar['name']}/points", 5)}
    node.create_subscription(Float32MultiArray, sensors.COMMAND_TOPIC, lambda m: on_command(state, m), 1)
    settings.set(GAMEPAD_CAMERA_SETTING, False)  # the pad drives the arm (via ROS), not the viewport camera
    state["task"] = asyncio.ensure_future(forward_sensors(state))
    state["drive_task"] = asyncio.ensure_future(drive_joints(state))
    print(f"[ROS2 bridge] Ready{' (with lidars)' if LIDARS else ''}. Press Play.")
    return state


stop_ros2_bridge()
ros2_bridge_state = start_ros2_bridge()
