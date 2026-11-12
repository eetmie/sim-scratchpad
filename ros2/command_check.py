#!/usr/bin/env python3
"""Smoke check of /excavator/command: +0.5 lift for 2 s moves revolute_lift; then the 150 ms timeout holds it."""

import time

import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

rclpy.init()
node = rclpy.create_node("command_check")
pub = node.create_publisher(Float32MultiArray, "/excavator/command", 1)
latest = {}
node.create_subscription(JointState, "/joint_states", lambda m: latest.update(zip(m.name, m.position)), 1)


def lift_after(seconds, command=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if command is not None:
            pub.publish(Float32MultiArray(data=command))
        rclpy.spin_once(node, timeout_sec=0.02)
    return latest.get("revolute_lift")


start = lift_after(2.0)
moved = lift_after(2.0, [0.0, 0.5, 0.0, 0.0])
held = lift_after(0.5)
after = lift_after(2.0)
ok = start is not None and abs(moved - start) > 0.01 and abs(after - held) < 0.005
verdict = "PASS" if ok else "FAIL"
print(f"{verdict} /excavator/command: lift {start} -> {moved} (commanded), then {held} -> {after} (stale)")
raise SystemExit(0 if ok else 1)
