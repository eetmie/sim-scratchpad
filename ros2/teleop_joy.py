#!/usr/bin/env python3
"""Gamepad -> /excavator/command, the same topic a VLA would publish. Sim and robot.

Needs the joy package's game_controller_node (SDL axis order); teleop.launch.py starts both:
    ros2 launch scripts/sim-scratchpad/ros2/teleop.launch.py

ISO excavator layout: left stick X swing, left stick Y arm (tilt), right stick X bucket, right stick Y boom (lift).
Streams at 50 Hz, zeros when the sticks are centred or the pad is gone, so the 150 ms command timeout only
trips when this node dies. On the robot, pump permission stays with the robot's own pad.
"""

import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensors  # noqa: E402

# [slew, lift, tilt, scoop] <- (joy axis, scale). SDL/joy axes: 0 left X, 1 left Y, 2 right X, 3 right Y;
# left and up are +1. Scales give the old in-Kit teleop's stick directions under sensors.COMMAND_SIGN.
STICKS = ((0, 1.0), (3, -1.0), (1, 1.0), (2, 1.0))


def deadzone(value, width):
    """Zero small stick noise, then rescale so the output still spans -1..1."""
    magnitude = (abs(value) - width) / (1.0 - width)
    return 0.0 if magnitude <= 0.0 else min(magnitude, 1.0) * (1.0 if value > 0 else -1.0)


class TeleopJoy(Node):
    def __init__(self):
        super().__init__("teleop_joy")
        self.deadzone = self.declare_parameter("deadzone", 0.2).value
        self.axes, self.stamp = None, None
        self.pub = self.create_publisher(Float32MultiArray, sensors.COMMAND_TOPIC, 1)
        self.create_subscription(Joy, "joy", self.on_joy, 1)
        self.create_timer(0.02, self.publish)

    def on_joy(self, msg):
        self.axes, self.stamp = msg.axes, self.get_clock().now()

    def publish(self):
        command = [0.0] * len(STICKS)
        fresh = self.stamp is not None and (self.get_clock().now() - self.stamp).nanoseconds < 0.5e9
        if fresh:  # joy republishes on change only; game_controller_node's autorepeat keeps it fresh
            command = [deadzone(self.axes[i], self.deadzone) * k if i < len(self.axes) else 0.0 for i, k in STICKS]
        self.pub.publish(Float32MultiArray(data=command))


def main():
    rclpy.init()
    rclpy.spin(TeleopJoy())


if __name__ == "__main__":
    main()
