#!/usr/bin/env python3
"""End-to-end check of depth_to_cloud's arm filter on a synthetic cabin depth image.

Renders the URDF's boom/arm/bucket surface over flat ground into the cabin camera, at a few arm poses, runs
robot_state_publisher, the camera TFs and depth_to_cloud with mapping.launch.py's parameters, and checks that arm
pixels come back in points_self and ground pixels away from the arm in points.

  source /opt/ros/<distro>/setup.bash && source ros2/ws/install/setup.bash && python3 ros2/test_arm_filter.py
"""

import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import yaml
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("mapping_launch", os.path.join(HERE, "mapping.launch.py"))
mapping_launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mapping_launch)
import arm_grid  # noqa: E402
import sensors  # noqa: E402
from robot_boxes import quat_matrix, read_urdf  # noqa: E402

W, H = sensors.DEPTH_SIZE
GROUND_Z = -0.08  # upper_carriage z of the floor: world_fixed puts lower_carriage 8 cm up, the slew joint at 0
POSES = {  # lift, tilt, tool [rad], inside the URDF limits; arm out in front of the cabin camera
    "reach": (0.0, 1.2, -0.6),
    "low": (0.35, 1.6, -1.2),
    "tucked": (-0.4, 2.2, -1.6),
}


def axis_rotation(axis, q):
    a = np.asarray(axis, float)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def arm_points(urdf_path, angles):
    """URDF arm surface points in the upper-carriage frame at these joint angles."""
    root = ET.parse(urdf_path).getroot()
    pose, pending = {"upper_carriage": np.eye(4)}, list(root.findall("joint"))
    while pending:
        for j in list(pending):
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            if parent in pose:
                M = pose[parent] @ arm_grid._origin(j)
                if j.get("type") == "revolute":
                    R = np.eye(4)
                    R[:3, :3] = axis_rotation(
                        [float(v) for v in j.find("axis").get("xyz").split()], angles[j.get("name")]
                    )
                    M = M @ R
                pose[child] = M
                pending.remove(j)
            elif parent in ("world", "lower_carriage"):
                pending.remove(j)
    out = []
    for link in root.findall("link"):
        name, vis = link.get("name"), link.find("visual")
        if name in ("lift_boom", "tilt_boom", "tool_body", "bucket"):
            mesh = os.path.join(HERE, "urdf", "meshes", os.path.basename(vis.find("geometry/mesh").get("filename")))
            V, F = arm_grid._load_obj(mesh)
            M = pose[name] @ arm_grid._origin(vis)
            out.append(arm_grid._surface(V, F, spacing=0.001) @ M[:3, :3].T + M[:3, 3])
    return np.vstack(out)


def render(points_upper, cam):
    """z-buffer splat into the camera: depth [mm] and which source won each pixel (1 ground, 2 arm)."""
    _, parent, xyz, q, fx, fy = cam
    R = quat_matrix(*q)  # upper <- camera
    depth = np.full((H, W), np.inf)
    label = np.zeros((H, W), np.uint8)
    for points, tag in points_upper:
        c = (points - np.asarray(xyz)) @ R  # camera frame
        c = c[c[:, 2] > 0.05]
        u = np.round(fx * c[:, 0] / c[:, 2] + W / 2).astype(int)
        v = np.round(fy * c[:, 1] / c[:, 2] + H / 2).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        for uu, vv, z in zip(u[ok], v[ok], c[ok, 2]):
            if z < depth[vv, uu]:
                depth[vv, uu], label[vv, uu] = z, tag
    depth[~np.isfinite(depth)] = 0
    return np.round(depth * 1000).astype(np.uint16), label


def unproject_px(cloud, fx, fy):
    xyz = np.frombuffer(bytes(cloud.data), np.float32).reshape(-1, 3)
    u = np.round(fx * xyz[:, 0] / xyz[:, 2] + W / 2).astype(int)
    v = np.round(fy * xyz[:, 1] / xyz[:, 2] + H / 2).astype(int)
    return set(zip(v.tolist(), u.tolist()))


def main():
    urdf_path = os.path.join(HERE, "urdf", "excavator.urdf")
    cam = next(c for c in sensors.DEPTH_CAMERAS if c[0] == "cabin")
    params = mapping_launch.depth_to_cloud_params("")
    params.update(cameras=["cabin"], stride=2, max_rate_hz=100.0)
    procs = []
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "p.yaml"), "w") as f:
            yaml.safe_dump({"depth_to_cloud": {"ros__parameters": params}}, f)
        procs.append(
            subprocess.Popen(
                ["ros2", "run", "excavator_depth_to_cloud", "depth_to_cloud", "--ros-args", "--params-file", f.name]
            )
        )
        for n, p, xyz, q, *_ in sensors.DEPTH_CAMERAS:
            keys = ("--x", "--y", "--z", "--qx", "--qy", "--qz", "--qw", "--frame-id", "--child-frame-id")
            args = [str(a) for pair in zip(keys, (*xyz, *q, p, sensors.camera_frame(n))) for a in pair]
            procs.append(
                subprocess.Popen(
                    ["ros2", "run", "tf2_ros", "static_transform_publisher", *args],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        rclpy.init()
        node = rclpy.create_node("arm_filter_test")
        with open(os.path.join(tmp, "rsp.yaml"), "w") as g:
            yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {"robot_description": read_urdf(HERE)}}}, g)
        procs.append(
            subprocess.Popen(
                [
                    "ros2",
                    "run",
                    "robot_state_publisher",
                    "robot_state_publisher",
                    "--ros-args",
                    "--params-file",
                    g.name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        joints = node.create_publisher(JointState, "/joint_states", 10)
        info_pub = node.create_publisher(CameraInfo, "/excavator/cameras/cabin/depth/camera_info", 2)
        depth_pub = node.create_publisher(Image, "/excavator/cameras/cabin/depth/image_raw", 1)
        got = {}
        node.create_subscription(
            PointCloud2, "/excavator/cameras/cabin/points", lambda m: got.__setitem__("kept", m), 2
        )
        node.create_subscription(
            PointCloud2, "/excavator/cameras/cabin/points_self", lambda m: got.__setitem__("self", m), 2
        )
        gx, gy = np.meshgrid(np.arange(0.0, 1.0, 0.002), np.arange(-0.5, 0.5, 0.002))
        ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, GROUND_Z)])
        fx, fy = cam[4], cam[5]
        info = CameraInfo(width=W, height=H, k=[fx, 0.0, W / 2, 0.0, fy, H / 2, 0.0, 0.0, 1.0])
        info.header.frame_id = sensors.camera_frame("cabin")
        failed = False
        try:
            for name, (lift, tilt, tool) in POSES.items():
                angles = {"revolute_carriage": 0.0, "revolute_lift": lift, "revolute_tilt": tilt, "revolute_tool": tool}
                depth, label = render([(ground, 1), (arm_points(urdf_path, angles), 2)], cam)
                got.clear()
                deadline = time.time() + 20
                while "self" not in got and time.time() < deadline:
                    now = node.get_clock().now().to_msg()
                    joints.publish(
                        JointState(
                            header=info.header.__class__(stamp=now), name=list(angles), position=list(angles.values())
                        )
                    )
                    rclpy.spin_once(node, timeout_sec=0.1)
                    info.header.stamp = now
                    msg = Image(
                        header=info.header, height=H, width=W, encoding="16UC1", step=2 * W, data=depth.tobytes()
                    )
                    info_pub.publish(info)
                    time.sleep(0.15)  # let TF for this stamp arrive first
                    depth_pub.publish(msg)
                    for _ in range(10):
                        rclpy.spin_once(node, timeout_sec=0.05)
                if "self" not in got:
                    print(f"{name}: no clouds from depth_to_cloud")
                    failed = True
                    continue
                kept, self_px = unproject_px(got["kept"], fx, fy), unproject_px(got["self"], fx, fy)
                sampled = np.zeros_like(label, bool)
                sampled[::2, ::2] = True
                arm = {(int(v), int(u)) for v, u in np.argwhere(sampled & (label == 2))}
                gnd = {(int(v), int(u)) for v, u in np.argwhere(sampled & (label == 1))}
                # Ground in range, clear of every margin (12 cm from the arm surface) and of the silhouette growth
                # (24 px from the arm's image; the bucket's filled hull reaches past its open mesh) must survive.
                near = np.zeros_like(label, bool)
                ys, xs = np.nonzero(label == 2)
                for dv in range(-24, 25, 2):
                    for du in range(-24, 25, 2):
                        near[np.clip(ys + dv, 0, H - 1), np.clip(xs + du, 0, W - 1)] = True
                R, t0 = quat_matrix(*cam[3]), np.asarray(cam[2])
                arm_s = arm_points(urdf_path, angles)[::10]
                far_gnd = set()
                for v, u in gnd:
                    if near[v, u]:
                        continue
                    z = depth[v, u] / 1000.0
                    c = np.array([(u - W / 2) / fx * z, (v - H / 2) / fy * z, z])
                    if (
                        np.linalg.norm(c) < 0.97 * sensors.DEPTH_MAX_M
                        and np.min(np.sum((arm_s - (R @ c + t0)) ** 2, 1)) > 0.12**2
                    ):
                        far_gnd.add((v, u))
                arm_cut = len(arm & self_px) / max(len(arm), 1)
                arm_leak = len(arm & kept) / max(len(arm), 1)
                far_kept = len(far_gnd & kept) / max(len(far_gnd), 1)
                print(
                    f"{name}: arm px {len(arm)}, cut {arm_cut:.3f}, leaked {arm_leak:.4f}; "
                    f"far ground px {len(far_gnd)}, kept {far_kept:.3f}"
                )
                if len(arm) < 200 or arm_leak > 0.002 or far_kept < 0.99:
                    failed = True
        finally:
            for p in procs:
                p.terminate()
            node.destroy_node()
            rclpy.shutdown()
    print("FAIL" if failed else "ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
