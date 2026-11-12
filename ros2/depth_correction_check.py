#!/usr/bin/env python3
"""Check depth_to_cloud's depth correction against sand_height.py's formula (numpy copy below), no sim needed.

    source /opt/ros/lyrical/setup.bash && source scripts/sim-scratchpad/ros2/ws/install/setup.bash
    python3 scripts/sim-scratchpad/ros2/depth_correction_check.py

Runs the C++ node on a synthetic depth image with random coefficients, plain and turned 180 deg, and compares
every output point with the reference. Exit code 0 = PASS.
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import StaticTransformBroadcaster

W, H, FX, CX, CY, STRIDE = 640, 480, 384.0, 325.5, 236.2, 2


def depth_basis(u, v, w, h, degree):  # sand_height.py, verbatim
    x, y = (u - w / 2) / (w / 2), (v - h / 2) / (w / 2)
    return np.stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)], -1)


def expected(depth_mm, coeffs, rotated):
    v, u = np.mgrid[0:H, 0:W].astype(float)
    if rotated:
        u, v = W - 1 - u, H - 1 - v
    k = depth_basis(u, v, W, H, 3) @ coeffs
    z = depth_mm * 0.001
    z = z / (1 - z * k)
    z, (vv, uu) = z[::STRIDE, ::STRIDE], np.mgrid[0:H:STRIDE, 0:W:STRIDE]
    return np.stack([(uu - CX) / FX * z, (vv - CY) / FX * z, z], -1).reshape(-1, 3)


def run(rotated, coeffs, depth_mm):
    params = {
        "depth_to_cloud": {
            "ros__parameters": {
                "cameras": ["t"],
                "max_range_m": 10.0,
                "box_links": ["t_link"],
                "box_data": [
                    100.0,
                    100.0,
                    100.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.01,
                    0.01,
                    0.01,
                ],  # far: drops nothing
                "t.correction_degree": 3,
                "t.correction_coeffs": coeffs.tolist(),
                "t.rotated_180": rotated,
            }
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(params, f)
    node_proc = subprocess.Popen(
        ["ros2", "run", "excavator_depth_to_cloud", "depth_to_cloud", "--ros-args", "--params-file", f.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    node = rclpy.create_node("depth_correction_check")
    got = []
    node.create_subscription(PointCloud2, "/excavator/cameras/t/points", got.append, 1)
    info_pub = node.create_publisher(CameraInfo, "/excavator/cameras/t/depth/camera_info", 2)
    depth_pub = node.create_publisher(Image, "/excavator/cameras/t/depth/image_raw", 2)
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id, tf.transform.rotation.w = "t_link", "t_cam", 1.0
    StaticTransformBroadcaster(node).sendTransform(tf)
    try:
        end = time.monotonic() + 20
        while not got and time.monotonic() < end:
            stamp = node.get_clock().now().to_msg()
            info_pub.publish(CameraInfo(width=W, height=H, k=[FX, 0, CX, 0, FX, CY, 0, 0, 1]))
            depth_pub.publish(
                Image(
                    header=tf.header.__class__(stamp=stamp, frame_id="t_cam"),
                    height=H,
                    width=W,
                    encoding="16UC1",
                    step=2 * W,
                    data=depth_mm.astype("<u2").tobytes(),
                )
            )
            rclpy.spin_once(node, timeout_sec=0.3)
    finally:
        node_proc.terminate()
        node.destroy_node()
    if not got:
        return None
    return np.frombuffer(got[0].data, np.float32).reshape(-1, 3)


def main():
    rclpy.init()
    rng = np.random.default_rng(1)
    v, u = np.mgrid[0:H, 0:W]
    depth_mm = (450 + 0.3 * u + 0.2 * v).astype(np.uint16)  # a tilted plane, 0.45-0.74 m
    coeffs = rng.normal(0, 0.03, 10)  # ~cm-level corrections at the edges, like the real fit
    ok = True
    for rotated in (False, True):
        points = run(rotated, coeffs, depth_mm)
        ref = expected(depth_mm, coeffs, rotated)
        if points is None or points.shape != ref.shape:
            print(f"FAIL rotated={rotated}: got {None if points is None else points.shape}, want {ref.shape}")
            ok = False
            continue
        err = np.abs(points - ref).max()
        shift = np.abs(ref[:, 2] - (depth_mm[::STRIDE, ::STRIDE].reshape(-1) * 0.001)).max()
        print(
            f"{'PASS' if err < 1e-5 else 'FAIL'} rotated={rotated}: max |C++ - reference| {err * 1000:.4f} mm "
            f"(correction moves points up to {shift * 1000:.1f} mm)"
        )
        ok &= err < 1e-5
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    main()
