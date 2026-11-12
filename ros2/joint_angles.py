#!/usr/bin/env python3
"""Sim only: /joint_states -> /excavator/joint_angles + /excavator/feedback_healthy, as kaivuri_bridge.py
publishes them on the robot (same normalization; out-of-bounds state is withheld, not clipped)."""

import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensors  # noqa: E402


def normalized(urdf_positions):
    q = np.asarray(urdf_positions, float) - sensors.URDF_OFFSET
    q[0] = math.remainder(q[0], 2 * math.pi)
    lo, hi = np.asarray(sensors.JOINT_ANGLE_MIN), np.asarray(sensors.JOINT_ANGLE_MAX)
    return 2 * (np.asarray(sensors.NATIVE_ANGLE_SIGN) * q - lo) / (hi - lo) - 1


class JointAngles(Node):
    def __init__(self):
        super().__init__("joint_angles")
        self.angles = self.create_publisher(Float32MultiArray, sensors.JOINT_ANGLES_TOPIC, 10)
        self.healthy = self.create_publisher(Bool, "/excavator/feedback_healthy", 10)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)

    def on_joints(self, msg):
        position = dict(zip(msg.name, msg.position))
        if any(name not in position for name in sensors.COMMAND_JOINTS):
            return
        v = normalized([position[name] for name in sensors.COMMAND_JOINTS])
        ok = bool((np.abs(v) <= 1).all())
        self.healthy.publish(Bool(data=ok))
        if ok:
            out = Float32MultiArray(data=v.astype(np.float32).tolist())
            out.layout.dim = [MultiArrayDimension(label=",".join(sensors.AXES), size=4, stride=4)]
            self.angles.publish(out)
        else:
            self.get_logger().warning(f"angles out of bounds: {np.round(v, 2)}", throttle_duration_sec=5.0)


def main():
    rclpy.init()
    rclpy.spin(JointAngles())


if __name__ == "__main__":
    main()
