"""The Newton sim as the robot on the contract's ROS 2 topics (sensor_contract.py), stamped in simulation time.

In:  /excavator/command            [slew, lift, tilt, scoop] valve demand in -1..1; stale after 150 ms = stop
Out: /clock, /joint_states         every physics step; raw joint angles [rad] are the state (the normalized
                                   /excavator/joint_angles is not published: ../ros2/joint_angles.py derives it)
     /excavator/cameras/<name>/image_raw   rgb8
     /excavator/heightmap, /excavator/heightmap/image   32FC1 [m] and mono8, the ground-truth map (on the robot
                                   the native depth mapper publishes these)
     /excavator/episode            JSON: {"event": "start", task, layout...} / {"event": "end", success, ...}
     /excavator/command            only when the sim drives itself (--auto_dig), so a bag always holds the actions

rclpy comes from Isaac Sim's bundled Jazzy (run_ros.sh sets it up); no Kit is involved.
"""

import json
import math
import time

import numpy as np
import rclpy
import sensor_contract as contract
from lidar import map_pixels
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, Header, String


class RosBridge:
    def __init__(self, joint_axes, joint_names, cameras):
        """``joint_axes``: the demo's joint -> (..., speed deg/s at command 1) table, whose commands this converts."""
        rclpy.init()
        self.node = rclpy.create_node("newton_sim")
        publisher = self.node.create_publisher
        self.clock = publisher(Clock, "/clock", 10)
        self.joints = publisher(JointState, "/joint_states", 10)
        self.command = publisher(Float32MultiArray, contract.COMMAND_TOPIC, 1)
        # Latched: a node that joins late (a policy runner) still learns the running episode and its task.
        latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.episode = publisher(String, contract.EPISODE_TOPIC, latched)
        self.images = {name: publisher(Image, contract.camera_topic(name, "image_raw"), 10) for name in cameras}
        self.height = publisher(Image, contract.HEIGHTMAP_TOPIC, 10)
        self.height_image = publisher(Image, contract.HEIGHTMAP_IMAGE_TOPIC, 10)
        # Depth 10: with depth 1, Fast DDS LARGE_DATA delivered big reliable messages at a fraction of their rate
        # (policy_node.py, 2026-11-17); a starved command subscription would trip the timeout and stutter the arm.
        self.node.create_subscription(Float32MultiArray, contract.COMMAND_TOPIC, self.on_command, 10)
        self.joint_names, self.received, self.open = list(joint_names), (0.0, np.zeros(len(contract.AXES))), None
        # Demo command (joint rate / its table speed) per unit of contract command, in the contract's axis order.
        speed = np.asarray(contract.COMMAND_SIGN) * np.asarray(contract.COMMAND_SPEED_DEG_S)
        self.scale = speed / np.array([joint_axes[name][3] for name in contract.COMMAND_JOINTS])
        # A recorder started before the sim must be listening before the first episode starts. `ros2 bag record
        # --use-sim-time` subscribes to nothing until it has seen a nonzero /clock, so hold the clock at 1 ns meanwhile.
        deadline = time.monotonic() + 5.0  # discovery across ROS distros took 2.3 s here
        while self.episode.get_subscription_count() == 0 and time.monotonic() < deadline:
            self.clock.publish(Clock(clock=self.stamp(1e-9).stamp))
            time.sleep(0.05)
        time.sleep(0.3)  # its other subscriptions match in the same burst

    def on_command(self, msg):
        """Keep the latest valid command; anything else is ignored, so the timeout stops the arm."""
        if len(msg.data) == len(contract.AXES) and all(math.isfinite(v) for v in msg.data):
            self.received = (time.monotonic(), np.clip(np.asarray(msg.data, float), -1, 1))

    def commands(self, own=None):
        """The demo's joint -> command dict. ``own``: the sim's scripted commands, published instead of listening."""
        rclpy.spin_once(self.node, timeout_sec=0)
        if own is not None:
            values = [own[name] / k for name, k in zip(contract.COMMAND_JOINTS, self.scale)]
            self.command.publish(Float32MultiArray(data=values))
            return own
        stamp, command = self.received
        # Wall clock: the timeout guards against a dead sender, however fast the sim runs.
        if time.monotonic() - stamp > contract.COMMAND_TIMEOUT_S:
            command = np.zeros(len(contract.AXES))
        return dict(zip(contract.COMMAND_JOINTS, command * self.scale))

    def limits(self, limits):
        """Joint limits [rad] narrowed to the robot's joint_angles bounds, kept 1 deg inside (slew stays free)."""
        limits = np.array(limits)
        for i, name in enumerate(contract.COMMAND_JOINTS):
            ends = [
                b / contract.NATIVE_ANGLE_SIGN[i] + contract.URDF_OFFSET[i]
                for b in (contract.JOINT_ANGLE_MIN[i], contract.JOINT_ANGLE_MAX[i])
            ]
            if max(ends) - min(ends) < 2 * math.pi - 1e-6:
                j = self.joint_names.index(name)
                limits[j] = max(limits[j, 0], min(ends) + math.radians(1)), min(limits[j, 1], max(ends) - math.radians(1))
        return limits

    @staticmethod
    def stamp(sim_time):
        sec, nanosec = divmod(round(sim_time * 1e9), 1_000_000_000)
        header = Header()
        header.stamp.sec, header.stamp.nanosec = sec, nanosec
        return header

    def state(self, sim_time, positions):
        header = self.stamp(sim_time)
        self.clock.publish(Clock(clock=header.stamp))
        self.joints.publish(JointState(header=header, name=self.joint_names, position=[float(p) for p in positions]))

    def image(self, publisher, sim_time, frame, pixels, encoding):
        header = self.stamp(sim_time)
        header.frame_id = frame
        height, width = pixels.shape[:2]
        publisher.publish(
            Image(
                header=header,
                height=height,
                width=width,
                encoding=encoding,
                is_bigendian=0,
                step=pixels.strides[0],
                data=pixels.tobytes(),
            )
        )

    def cameras(self, sim_time, rgb_images):
        for name, publisher in self.images.items():
            self.image(publisher, sim_time, contract.camera_frame(name), np.ascontiguousarray(rgb_images[name]), "rgb8")

    def heightmap(self, sim_time, height):
        self.image(self.height, sim_time, contract.HEIGHTMAP_FRAME, height.astype(np.float32), "32FC1")
        self.image(self.height_image, sim_time, contract.HEIGHTMAP_FRAME, map_pixels(height), "mono8")

    def end_episode(self, sim_time, result):
        """Close the running episode with its ``result`` (must hold ``success``)."""
        if self.open is not None:
            event = {"event": "end", "episode": self.open, "time_s": round(sim_time, 4), **result}
            self.episode.publish(String(data=json.dumps(event)))

    def start_episode(self, sim_time, layout):
        self.open = 0 if self.open is None else self.open + 1
        event = {"event": "start", "episode": self.open, "time_s": round(sim_time, 4), **layout}
        self.episode.publish(String(data=json.dumps(event)))

    def close(self):
        time.sleep(0.2)  # let the last reliable messages (the episode end) leave before the node goes
        self.node.destroy_node()
        rclpy.try_shutdown()
