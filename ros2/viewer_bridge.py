"""Standard ROS 2 publishers for processed viewer observations. No Isaac/Kit/USD imports.

The caller supplies arrays, poses, and acquisition timestamps. Depth and IR are opt-in.
Only standard ROS messages are required, including the FastDEM map's PointCloud2 view.
"""

import json
import math

import numpy as np
from ros2.map_contract import FRONT_FRAME, SURROUND_FRAME, depth_millimetres, grid_points, height_image, height_metres
from scipy.spatial.transform import Rotation


class ViewerRosBridge:
    def __init__(self, profile, *, clock_mode, cameras=False, depth=False, cloud_hz=None):
        # Import only after the launcher's ROS environment/Isaac bridge is ready.
        import rclpy
        from builtin_interfaces.msg import Time
        from geometry_msgs.msg import TransformStamped
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from rclpy.signals import SignalHandlerOptions
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField, TimeReference
        from std_msgs.msg import Header, String
        from tf2_ros import TransformBroadcaster

        if cloud_hz is not None and (not math.isfinite(cloud_hz) or cloud_hz < 0):
            raise ValueError("ROS cloud frequency must be finite and nonnegative")
        self.profile, self.cloud_hz = profile, cloud_hz
        self.types = dict(
            Time=Time,
            TransformStamped=TransformStamped,
            Clock=Clock,
            CameraInfo=CameraInfo,
            Image=Image,
            JointState=JointState,
            PointCloud2=PointCloud2,
            PointField=PointField,
            TimeReference=TimeReference,
            Header=Header,
            String=String,
        )
        self.context = Context()
        rclpy.init(args=[], context=self.context, signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node("isaac_terrain", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.tf = TransformBroadcaster(self.node)
        self.publishers = {}
        self.last_cloud_ns = None
        self.last_stamp_ns = None
        self.episode = None
        self.clock_mode = clock_mode

        def publisher(key, message, topic, qos=1):
            self.publishers[key] = self.node.create_publisher(message, topic, qos)

        publisher("clock", Clock, "/clock")
        publisher("joints", JointState, "/joint_states", 10)
        publisher("host_time", TimeReference, "/excavator/time/host_capture")
        publisher("host_steady", TimeReference, "/excavator/time/host_monotonic")
        publisher("source_time", TimeReference, "/excavator/time/source_capture")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        publisher("metadata", String, "/excavator/maps/metadata", latched)
        publisher(
            "episode", String, "/excavator/episode", QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        for key, topic in (
            ("front", "/excavator/heightmap"),
            ("front_image", "/excavator/heightmap/image"),
            ("surround", "/fastdem/mapping/height"),
            ("surround_image", "/fastdem/mapping/image"),
            ("age", "/fastdem/mapping/age"),
        ):
            publisher(key, Image, topic)
        if cloud_hz != 0:
            self.node.declare_parameter("publish_map_cloud", True)
            publisher("cloud", PointCloud2, "/fastdem/mapping/cloud")
        for name in ("cabin", "arm"):
            prefix = f"/excavator/cameras/{name}"
            if cameras:
                publisher(f"ir_{name}", Image, prefix + "/image_raw")
                publisher(f"ir_info_{name}", CameraInfo, prefix + "/camera_info")
            if depth:
                publisher(f"depth_{name}", Image, prefix + "/depth/image_raw")
                publisher(f"depth_info_{name}", CameraInfo, prefix + "/depth/camera_info")
        metadata = {
            "backend": profile["mapping_backend"],
            "robot_revision": profile.get("robot_revision"),
            "front": {**profile["front"], "frame_id": FRONT_FRAME},
            "surround": {**profile["surround"], "frame_id": SURROUND_FRAME},
            "height_units": "metres above registered sand reference; NaN unknown",
            "layout": "row decreases X; column decreases Y; cell centres are half a cell inside bounds",
            "gray": {
                "zero_height": 128,
                "limit_m": profile["height_color_limit_m"],
                "negative_limit": 16,
                "positive_limit": 240,
                "unknown": 0,
            },
            "age_units": profile.get("map_age_units", "nominal dataset frames"),
            "age_unknown": 65535,
            "clock_mode": clock_mode,
            "clock_epoch_s": 1,
            "time_reference": (
                "headers use /clock; host_capture is Unix UTC; host_monotonic is steady host time; "
                "source_capture is recorded host monotonic"
            ),
            "raw_depth": "16UC1 millimetres; unrotated physical optical frame; zero invalid" if depth else "disabled",
        }
        self.publishers["metadata"].publish(String(data=json.dumps(metadata)))
        self.clock(1_000_000_000)

    def time(self, stamp_ns):
        sec, nanosec = divmod(int(stamp_ns), 1_000_000_000)
        return self.types["Time"](sec=sec, nanosec=nanosec)

    def header(self, stamp_ns, frame=""):
        return self.types["Header"](stamp=self.time(stamp_ns), frame_id=frame)

    def clock(self, stamp_ns):
        self.publishers["clock"].publish(self.types["Clock"](clock=self.time(stamp_ns)))

    def spin(self):
        self.executor.spin_once(timeout_sec=0)

    def image(self, key, pixels, encoding, stamp_ns, frame):
        pixels = np.ascontiguousarray(pixels)
        self.publishers[key].publish(
            self.types["Image"](
                header=self.header(stamp_ns, frame),
                height=pixels.shape[0],
                width=pixels.shape[1],
                encoding=encoding,
                is_bigendian=0,
                step=pixels.strides[0],
                data=pixels.tobytes(),
            )
        )

    def camera_info(self, key, intr, stamp_ns, frame):
        fx, fy, cx, cy = (float(intr[k]) for k in ("fx", "fy", "cx", "cy"))
        self.publishers[key].publish(
            self.types["CameraInfo"](
                header=self.header(stamp_ns, frame),
                width=int(intr["width"]),
                height=int(intr["height"]),
                distortion_model="plumb_bob",
                d=[float(c) for c in intr["coeffs"]],
                k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
                r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                p=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
        )

    def event(self, event, episode, stamp_ns):
        payload = {
            "event": event,
            "episode": episode,
            "time_s": stamp_ns * 1e-9,
            "clock_mode": self.clock_mode,
            "task": "calibrated_scene_replay",
        }
        self.publishers["episode"].publish(self.types["String"](data=json.dumps(payload)))

    def publish(self, arrays, state, *, stamp_ns, episode=0, source_ns=None):
        self.publish_exposure(arrays, state, stamp_ns=stamp_ns, episode=episode, source_ns=source_ns)
        self.publish_maps(arrays, stamp_ns)

    def publish_exposure(self, arrays, state, *, stamp_ns, episode, source_ns):
        if self.node.count_publishers("/clock") > 1:
            raise RuntimeError("Another /clock publisher is active. Use a separate ROS_DOMAIN_ID for this run.")
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            raise ValueError("Observation timestamps must increase; never mix or reset the ROS clock")
        if episode != self.episode:
            if self.episode is not None:
                self.event("end", self.episode, self.last_stamp_ns)
            self.episode = episode
            self.event("start", episode, stamp_ns)
        self.last_stamp_ns = stamp_ns
        self.clock(stamp_ns)
        for key, reference, source in (
            ("host_time", int(arrays["capture_host_unix_ns"]), "host_unix_utc"),
            ("host_steady", int(arrays["capture_host_monotonic_ns"]), "host_monotonic"),
            ("source_time", source_ns, "recorded_host_monotonic"),
        ):
            if reference is not None:
                self.publishers[key].publish(
                    self.types["TimeReference"](
                        header=self.header(stamp_ns), time_ref=self.time(reference), source=source
                    )
                )
        transforms = []
        for parent, child, matrix in state["transforms"]:
            transform = self.types["TransformStamped"](header=self.header(stamp_ns, parent), child_frame_id=child)
            p, q = matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = (
                map(float, p)
            )
            (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ) = map(float, q)
            transforms.append(transform)
        self.tf.sendTransform(transforms)
        self.publishers["joints"].publish(
            self.types["JointState"](
                header=self.header(stamp_ns), name=list(state["joints"]), position=list(state["joints"].values())
            )
        )
        for name, (pixels, intr) in state["images"].items():
            frame = f"{name}_ir_optical"
            self.image(f"ir_{name}", pixels, "mono8", stamp_ns, frame)
            self.camera_info(f"ir_info_{name}", intr, stamp_ns, frame)

    def publish_maps(self, arrays, stamp_ns):
        limit = self.profile["height_color_limit_m"]
        for key, frame in (("front", FRONT_FRAME), ("surround", SURROUND_FRAME)):
            self.image(key, height_metres(arrays[f"{key}_mm"]), "32FC1", stamp_ns, frame)
            self.image(key + "_image", height_image(arrays[f"{key}_mm"], limit), "mono8", stamp_ns, frame)
        age = arrays.get("surround_age_updates", arrays["surround_age_frames"])
        self.image("age", np.asarray(age, dtype="<u2"), "16UC1", stamp_ns, SURROUND_FRAME)
        cloud = self.publishers.get("cloud")
        if (
            cloud is not None
            and cloud.get_subscription_count()
            and self.node.get_parameter("publish_map_cloud").value
            and (
                self.cloud_hz is None
                or self.last_cloud_ns is None
                or stamp_ns - self.last_cloud_ns >= 1e9 / self.cloud_hz
            )
        ):
            geometry = self.profile["surround"]
            points = grid_points(
                height_metres(arrays["surround_mm"]), geometry["xy_m"], geometry["xy_m"], geometry["cell_m"]
            )
            field = self.types["PointField"]
            cloud.publish(
                self.types["PointCloud2"](
                    header=self.header(stamp_ns, SURROUND_FRAME),
                    height=1,
                    width=len(points),
                    fields=[
                        field(name=name, offset=4 * i, datatype=field.FLOAT32, count=1) for i, name in enumerate("xyz")
                    ],
                    is_bigendian=False,
                    point_step=12,
                    row_step=len(points) * 12,
                    data=points.tobytes(),
                    is_dense=True,
                )
            )
            self.last_cloud_ns = stamp_ns
        for name in ("cabin", "arm"):
            if f"depth_{name}" not in self.publishers:
                continue
            sensor = self.profile["sensors"][name]
            frame = f"{name}_camera"
            if self.publishers[f"depth_{name}"].get_subscription_count():
                self.image(
                    f"depth_{name}", depth_millimetres(arrays[f"depth_{name}_z16"], sensor), "16UC1", stamp_ns, frame
                )
            if not self.publishers[f"depth_info_{name}"].get_subscription_count():
                continue
            intr = {
                **sensor["depth_intrinsics"],
                "cx": sensor["depth_intrinsics"]["ppx"],
                "cy": sensor["depth_intrinsics"]["ppy"],
            }
            self.camera_info(f"depth_info_{name}", intr, stamp_ns, frame)

    def close(self):
        if self.episode is not None:
            from rclpy.duration import Duration

            self.event("end", self.episode, self.last_stamp_ns)
            self.publishers["episode"].wait_for_all_acked(Duration(seconds=0.5))
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.try_shutdown()
