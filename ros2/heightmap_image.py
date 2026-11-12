#!/usr/bin/env python3
"""FastDEM's latest scan -> a top-down height image in the cabin frame, for the VLA's vision encoder. Sim and robot.

Subscribes /fastdem/scan/rasterized (one point per map cell the scan saw, frame "world", stamped with the scan),
moves it into HEIGHTMAP_FRAME with TF at the scan stamp and bins it into a fixed grid in front of the cabin, so the
image turns with the swing like the camera. Publishes sensors.HEIGHTMAP_TOPIC (32FC1, m) and
HEIGHTMAP_IMAGE_TOPIC (mono8). Only what this scan saw: the remembered map is FastDEM's /fastdem/mapping/*.
"""

import os
import sys

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensors  # noqa: E402
from robot_boxes import quat_matrix  # noqa: E402


def rasterize(points):
    """Highest point per cell; NaN where nothing landed. points: (N, 3) in HEIGHTMAP_FRAME."""
    n, (x0, x1), (y0, y1) = sensors.HEIGHTMAP_SIZE, sensors.HEIGHTMAP_X_M, sensors.HEIGHTMAP_Y_M
    row = ((x1 - points[:, 0]) * n / (x1 - x0)).astype(int)  # row 0 = far edge
    col = ((y1 - points[:, 1]) * n / (y1 - y0)).astype(int)  # column 0 = left edge
    inside = (row >= 0) & (row < n) & (col >= 0) & (col < n) & np.isfinite(points[:, 2])
    height = np.full(n * n, -np.inf, dtype=np.float32)
    np.maximum.at(height, row[inside] * n + col[inside], points[inside, 2].astype(np.float32))
    height[np.isneginf(height)] = np.nan
    return height.reshape(n, n)


def to_mono8(height):
    low, high = sensors.HEIGHTMAP_Z_M
    gray = 1 + np.rint(254 * np.clip((height - low) / (high - low), 0, 1))
    return np.where(np.isnan(height), 0, gray).astype(np.uint8)


class HeightmapImage(Node):
    def __init__(self):
        super().__init__("heightmap_image")
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, None, spin_thread=True)  # own thread: lookups can wait
        self.height = self.create_publisher(Image, sensors.HEIGHTMAP_TOPIC, 2)
        self.image = self.create_publisher(Image, sensors.HEIGHTMAP_IMAGE_TOPIC, 2)
        self.create_subscription(PointCloud2, "/fastdem/scan/rasterized", self.on_scan, 1)

    def on_scan(self, msg):
        try:
            t = self.tf.lookup_transform(
                sensors.HEIGHTMAP_FRAME, msg.header.frame_id, Time.from_msg(msg.header.stamp), Duration(seconds=0.3)
            ).transform
        except TransformException as exc:
            self.get_logger().warning(f"no TF at the scan stamp: {exc}", throttle_duration_sec=5.0)
            return
        fields = {f.name: f.offset for f in msg.fields}
        raw = np.frombuffer(msg.data, np.uint8).reshape(-1, msg.point_step)
        points = np.stack([raw[:, fields[k] : fields[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], axis=1)
        q, p = t.rotation, t.translation
        height = rasterize(points @ quat_matrix(q.x, q.y, q.z, q.w).T + (p.x, p.y, p.z))
        header = Header(stamp=msg.header.stamp, frame_id=sensors.HEIGHTMAP_FRAME)
        n = sensors.HEIGHTMAP_SIZE
        self.height.publish(
            Image(header=header, height=n, width=n, encoding="32FC1", is_bigendian=0, step=4 * n, data=height.tobytes())
        )
        self.image.publish(
            Image(
                header=header,
                height=n,
                width=n,
                encoding="mono8",
                is_bigendian=0,
                step=n,
                data=to_mono8(height).tobytes(),
            )
        )


def main():
    rclpy.init()
    rclpy.spin(HeightmapImage())


if __name__ == "__main__":
    main()
