"""Publish this machine's mesh paths for remote RViz. The simulator owns all TF."""

from pathlib import Path

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from robot_boxes import read_urdf
from std_msgs.msg import String


def main():
    rclpy.init()
    node = rclpy.create_node("viewer_robot_description")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(String, "/robot_description", qos)
    publisher.publish(String(data=read_urdf(Path(__file__).resolve().parent.as_posix())))
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
