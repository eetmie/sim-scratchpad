"""Excavator teleop demo -> ROS 2, for viewing in RViz. Run in Isaac Sim's Script Editor, then press Play.

Start Isaac Sim with ros2/run_isaacsim.sh (sets up the bundled Jazzy ROS 2 libs), open demo_env.usd,
then run this file. It also starts excavator_teleop.py (which starts lidar_demo_script.py). Rerun to restart.

Published (all stamped with sim time; frames are in the URDF from export_urdf.py):
    /clock                                   rosgraph_msgs/Clock
    /joint_states                            sensor_msgs/JointState (rad), feeds robot_state_publisher
    /cam_cabin/image_raw, /cam_boom/...      sensor_msgs/Image rgb8 (+ camera_info)
    /lidar_cabin/points, /lidar_boom/points  sensor_msgs/PointCloud2, frame "world", excavator hits removed
    /excavator/heightmap                     sensor_msgs/Image 32FC1, height above ground in m, NaN = unseen
                                             row 0 = +X edge, column 0 = +Y edge (same as the omni.ui view)
    /excavator/heightmap_cloud               sensor_msgs/PointCloud2, one point per known cell, frame "world"
"""

import asyncio
import os
from pathlib import Path

import numpy as np

import carb
import omni.graph.core as og
import omni.kit.app
import omni.usd
from pxr import Sdf, Usd

TELEOP_SCRIPT = "/home/joel/IsaacLab/scripts/sim-scratchpad/excavator_teleop.py"
GRAPH_PATH = "/ROS2Bridge"
ROBOT_PATH = "/World/excavator_bucket"
# (camera prim, ROS name). Small images keep the extra render cost low.
CAMERAS = [
    ("/World/excavator_bucket/upper_carriage/cam_cabin", "cam_cabin"),
    ("/World/excavator_bucket/tilt_boom/cam_boom", "cam_boom"),
]
CAMERA_RES = (320, 240)
# Lidar debug points in the viewport also land in the camera images; RViz shows them instead.
SHOW_LIDAR_POINTS_IN_SIM = False
WORLD_FRAME = "world"
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
from sensor_msgs.msg import Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Header  # noqa: E402

from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

XYZ_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]


def build_graph(stage):
    """One action graph for clock, joint states and both cameras. Rebuilt from scratch on rerun.

    Lives on the session layer, so saving demo_env.usd never stores it.
    """
    keys = og.Controller.Keys
    nodes = [
        ("Tick", "omni.graph.action.OnPlaybackTick"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("ReadJoints", "isaacsim.sensors.physics.IsaacReadJointState"),
        ("PublishJoints", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
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
        ("Tick.outputs:tick", "RunOnce.inputs:execIn"),
        ("Context.outputs:context", "PublishJoints.inputs:context"),
        ("SimTime.outputs:simulationTime", "PublishJoints.inputs:timeStamp"),
    ]
    connects.append(("ReadJoints.outputs:execOut", "PublishJoints.inputs:execIn"))
    for field in JOINT_STATE_FIELDS:
        connects.append((f"ReadJoints.outputs:{field}", f"PublishJoints.inputs:{field}"))
    for prim, name in CAMERAS:
        render, rgb, info = f"Render_{name}", f"Rgb_{name}", f"Info_{name}"
        nodes += [
            (render, "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            (rgb, "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (info, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        values += [
            (f"{render}.inputs:cameraPrim", [Sdf.Path(prim)]),
            (f"{render}.inputs:width", CAMERA_RES[0]),
            (f"{render}.inputs:height", CAMERA_RES[1]),
            (f"{rgb}.inputs:type", "rgb"),
            (f"{rgb}.inputs:topicName", f"{name}/image_raw"),
            (f"{rgb}.inputs:frameId", name),
            (f"{rgb}.inputs:resetSimulationTimeOnStop", True),
            (f"{info}.inputs:topicName", f"{name}/camera_info"),
            (f"{info}.inputs:frameId", name),
            (f"{info}.inputs:resetSimulationTimeOnStop", True),
        ]
        connects.append(("RunOnce.outputs:step", f"{render}.inputs:execIn"))
        for helper in (rgb, info):
            connects += [
                (f"{render}.outputs:execOut", f"{helper}.inputs:execIn"),
                (f"{render}.outputs:renderProductPath", f"{helper}.inputs:renderProductPath"),
                ("Context.outputs:context", f"{helper}.inputs:context"),
            ]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        if stage.GetPrimAtPath(GRAPH_PATH):
            stage.RemovePrim(GRAPH_PATH)
        og.Controller.edit(
            {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
            {keys.CREATE_NODES: nodes, keys.SET_VALUES: values, keys.CONNECT: connects},
        )


def cloud_msg(header, points):
    points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        fields=XYZ_FIELDS,
        is_bigendian=False,
        point_step=12,
        row_step=12 * len(points),
        is_dense=True,
        data=points.tobytes(),
    )


def heightmap_msgs(header, memory):
    """32FC1 image of the map, plus a cloud of known cell centers (for the 3D view)."""
    height = memory["height_m"].astype(np.float32)
    rows, cols = height.shape
    image = Image(
        header=header, height=rows, width=cols, encoding="32FC1", is_bigendian=0, step=4 * cols, data=height.tobytes()
    )
    (x0, x1), (y0, y1) = HEIGHTMAP_X_M, HEIGHTMAP_Y_M  # noqa: F821 - from lidar_demo_script.py
    r, c = np.nonzero(memory["valid"])
    cells = np.stack(
        [
            x1 - (r + 0.5) * (x1 - x0) / rows,
            y1 - (c + 0.5) * (y1 - y0) / cols,
            height[r, c] + GROUND_Z_M,  # noqa: F821
        ],
        axis=1,
    )
    return image, cloud_msg(header, cells)


def publish_new_lidar_data(state, lidar):
    """Publish scans and the height map that changed since the last call."""
    # shortcut: stamps are the sim time at publish, up to one 5 Hz update after the scan; use the scan's own
    # timestamp if the model needs exact image/scan alignment.
    sim_time = SimulationManager.get_simulation_time()
    header = Header(frame_id=WORLD_FRAME)
    header.stamp.sec, header.stamp.nanosec = int(sim_time), int((sim_time % 1.0) * 1e9)
    for (path, _), sensor in zip(SENSORS, lidar["sensors"]):  # noqa: F821 - same order as lidar_demo_script.py
        name = path.rsplit("/", 1)[-1]
        scan = sensor.get("last_scan")
        if scan is None or scan == state["last_scans"].get(name) or sensor.get("kept") is None:
            continue
        state["last_scans"][name] = scan
        if name not in state["clouds"]:
            state["clouds"][name] = state["node"].create_publisher(PointCloud2, f"{name}/points", 5)
        state["clouds"][name].publish(cloud_msg(header, sensor["kept"]))
    memory = lidar["heightmap"]
    if memory.get("stamp_ns") != state["last_map"]:
        state["last_map"] = memory.get("stamp_ns")
        image, cloud = heightmap_msgs(header, memory)
        state["heightmap"].publish(image)
        state["heightmap_cloud"].publish(cloud)


async def forward_lidar(state):
    app = omni.kit.app.get_app()
    last_error = None
    while True:
        await app.next_update_async()
        lidar = globals().get("cabin_lidar_state")
        if lidar is None:
            continue
        try:
            publish_new_lidar_data(state, lidar)
        except Exception as exc:  # keep publishing; an unhandled error would end this task without a word
            if repr(exc) != last_error:
                carb.log_error(f"[ROS2 bridge] lidar/height map publish failed: {exc!r}")
                last_error = repr(exc)


def stop_ros2_bridge():
    state = globals().pop("ros2_bridge_state", None)
    if state is not None:
        state["task"].cancel()
        state["node"].destroy_node()


def start_ros2_bridge():
    build_graph(omni.usd.get_context().get_stage())
    if not rclpy.ok():
        rclpy.init()
    node = Node("excavator_sim_bridge")
    state = {
        "node": node,
        "clouds": {},
        "last_scans": {},
        "last_map": None,
        "heightmap": node.create_publisher(Image, "excavator/heightmap", 5),
        "heightmap_cloud": node.create_publisher(PointCloud2, "excavator/heightmap_cloud", 5),
    }
    state["task"] = asyncio.ensure_future(forward_lidar(state))
    print("[ROS2 bridge] Graph at /ROS2Bridge + height map / lidar publishers ready. Press Play.")
    return state


# Teleop first (it starts the lidar script), sharing this namespace so cabin_lidar_state is visible here.
exec(compile(Path(TELEOP_SCRIPT).read_text(), TELEOP_SCRIPT, "exec"), globals())
DRAW_LIDAR_POINTS = SHOW_LIDAR_POINTS_IN_SIM  # read by lidar_demo_script.draw_points at each redraw
stop_ros2_bridge()
ros2_bridge_state = start_ros2_bridge()
