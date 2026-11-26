"""Geometric lidar and cabin-frame maps. No ROS or RTX dependency."""

import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp
from sensor_contract import (
    DEPTH_CAMERAS,
    DEPTH_SIZE,
    HEIGHTMAP_SIZE,
    HEIGHTMAP_X_M,
    HEIGHTMAP_Y_M,
    HEIGHTMAP_Z_M,
    LIDARS,
)
from soil import pose_matrix


@wp.kernel
def cast_meshes(
    ground: wp.uint64,
    moving: wp.uint64,
    robot: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    near: float,
    far: float,
    distances: wp.array(dtype=float),
    labels: wp.array(dtype=int),
):
    i = wp.tid()
    origin = origins[i] + near * directions[i]
    distance = far - near
    label = int(0)
    a = wp.mesh_query_ray(ground, origin, directions[i], distance)
    if a.result:
        distance = a.t
        label = 1
    if moving != wp.uint64(0):
        b = wp.mesh_query_ray(moving, origin, directions[i], distance)
        if b.result:
            distance = b.t
            label = 2
    if robot != wp.uint64(0):
        c = wp.mesh_query_ray(robot, origin, directions[i], distance)
        if c.result:
            distance = c.t
            label = 3
    distances[i] = distance + near
    labels[i] = label


class TriangleScene:
    """Refit fixed topology; rebuild the small moving surface when its topology changes."""

    def __init__(self, device):
        self.device = device
        self.meshes = {}
        self.indices = {}

    def update(self, name, points, indices):
        indices = np.asarray(indices, dtype=np.int32).ravel()
        points = np.ascontiguousarray(points, dtype=np.float32)
        if not len(indices):
            self.meshes.pop(name, None)
            self.indices.pop(name, None)
            return
        old = self.meshes.get(name)
        if old is not None and len(old.points) == len(points) and np.array_equal(self.indices[name], indices):
            old.points.assign(points)
            old.refit()
        else:
            self.meshes[name] = wp.Mesh(
                points=wp.array(points, dtype=wp.vec3, device=self.device),
                indices=wp.array(indices, dtype=int, device=self.device),
            )
            self.indices[name] = indices.copy()

    def cast(self, origins, directions, near=0.01, far=0.8, include_robot=True):
        n = len(origins)
        distances = wp.empty(n, dtype=float, device=self.device)
        labels = wp.empty(n, dtype=int, device=self.device)
        wp.launch(
            cast_meshes,
            n,
            [
                self.meshes["ground"].id,
                self.meshes["moving"].id if "moving" in self.meshes else 0,
                self.meshes["robot"].id if include_robot and "robot" in self.meshes else 0,
                wp.array(origins, dtype=wp.vec3, device=self.device),
                wp.array(directions, dtype=wp.vec3, device=self.device),
                near,
                far,
                distances,
                labels,
            ],
            device=self.device,
        )
        return distances.numpy(), labels.numpy()


def fan(step_degrees):
    angle = np.deg2rad(np.linspace(-30, 30, round(60 / step_degrees) + 1))
    elevation, azimuth = np.meshgrid(angle, angle, indexing="ij")
    return (
        np.stack((np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)), axis=-1)
        .reshape(-1, 3)
        .astype(np.float32)
    )


def pinhole_rays(fx, fy, width, height):
    """Unit optical-frame rays through pixel centers, with a centered principal point."""
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays = np.stack(((u - width / 2) / fx, (v - height / 2) / fy, np.ones_like(u)), axis=-1).reshape(-1, 3)
    return (rays / np.linalg.norm(rays, axis=1, keepdims=True)).astype(np.float32)


def capture_depth(scene, origin, rotation, rays, width, height):
    directions = rays @ rotation.T
    starts = np.broadcast_to(origin, directions.shape).copy()
    distance, labels = scene.cast(starts, directions, near=0.05, far=2)
    depth = distance * rays[:, 2]  # optical Z rather than distance along the ray
    depth[labels == 0] = np.nan
    return depth.reshape(height, width)


def rasterize(world_points, cabin_pose):
    """Latest scan only: local cabin Z, row zero +X, column zero +Y; unknown stays NaN."""
    local = (world_points - cabin_pose[:3]) @ pose_matrix(cabin_pose)
    x0, x1 = HEIGHTMAP_X_M
    y0, y1 = HEIGHTMAP_Y_M
    valid = np.isfinite(local).all(axis=1) & (local[:, 0] >= x0) & (local[:, 0] < x1)
    valid &= (local[:, 1] >= y0) & (local[:, 1] < y1)
    p = local[valid]
    col = HEIGHTMAP_SIZE - 1 - ((p[:, 1] - y0) / (y1 - y0) * HEIGHTMAP_SIZE).astype(int)
    row = HEIGHTMAP_SIZE - 1 - ((p[:, 0] - x0) / (x1 - x0) * HEIGHTMAP_SIZE).astype(int)
    out = np.full((HEIGHTMAP_SIZE, HEIGHTMAP_SIZE), -np.inf, dtype=np.float32)
    np.maximum.at(out, (row, col), p[:, 2])
    out[~np.isfinite(out)] = np.nan
    return out


def ground_map(soil, height, cabin_pose):
    """``height`` (a soil grid [m]) on the cabin-frame crop, laid out like ``rasterize``; NaN outside the grid.

    Fed the displayed ground, this is the map of what lies on the ground: no robot, no carried or falling sand.
    """
    x = HEIGHTMAP_X_M[1] - (np.arange(HEIGHTMAP_SIZE) + 0.5) * np.ptp(HEIGHTMAP_X_M) / HEIGHTMAP_SIZE
    y = HEIGHTMAP_Y_M[1] - (np.arange(HEIGHTMAP_SIZE) + 0.5) * np.ptp(HEIGHTMAP_Y_M) / HEIGHTMAP_SIZE
    xx, yy = np.meshgrid(x, y, indexing="ij")
    rotation = pose_matrix(cabin_pose)
    world = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size))) @ rotation.T + cabin_pose[:3]
    f = (world[:, 0] - soil.origin) / soil.cfg.cell - 0.5
    g = (world[:, 1] - soil.origin) / soil.cfg.cell - 0.5
    inside = (f >= 0) & (g >= 0) & (f <= soil.n - 1) & (g <= soil.n - 1)
    i, j = np.clip(f.astype(int), 0, soil.n - 2), np.clip(g.astype(int), 0, soil.n - 2)
    a, b = np.clip(f - i, 0, 1), np.clip(g - j, 0, 1)
    world[:, 2] = (1 - b) * ((1 - a) * height[j, i] + a * height[j, i + 1]) + b * (
        (1 - a) * height[j + 1, i] + a * height[j + 1, i + 1]
    )
    local = ((world - cabin_pose[:3]) @ rotation)[:, 2]
    local[~inside] = np.nan
    return local.reshape(HEIGHTMAP_SIZE, HEIGHTMAP_SIZE).astype(np.float32)


def map_pixels(height):
    valid = np.isfinite(height)
    gray = np.zeros(height.shape, dtype=np.uint8)
    low, high = HEIGHTMAP_Z_M
    gray[valid] = 1 + np.rint(254 * np.clip((height[valid] - low) / (high - low), 0, 1)).astype(np.uint8)
    return gray


class RobotGeometry:
    """URDF visual triangles moved by Newton link poses, used only for ray occlusion."""

    def __init__(self, names, directory):
        # The same exported URDF meshes used by RViz and the real ROS self-filter.
        self.parts = []
        for link in ET.parse(directory / "excavator.urdf").getroot().findall("link"):
            if link.get("name") not in names:
                continue
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                path = directory / "meshes" / Path(mesh.get("filename")).name
                vertices, faces = [], []
                for line in path.read_text().splitlines():
                    if line.startswith("v "):
                        vertices.append([float(v) for v in line.split()[1:4]])
                    elif line.startswith("f "):
                        ids = [int(v.split("/")[0]) - 1 for v in line.split()[1:]]
                        for i in range(1, len(ids) - 1):
                            faces.extend((ids[0], ids[i], ids[i + 1]))
                p = np.asarray(vertices) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                origin = visual.find("origin")
                if origin is not None:
                    r, pitch, y = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") / 2
                    q = [
                        np.sin(r) * np.cos(pitch) * np.cos(y) - np.cos(r) * np.sin(pitch) * np.sin(y),
                        np.cos(r) * np.sin(pitch) * np.cos(y) + np.sin(r) * np.cos(pitch) * np.sin(y),
                        np.cos(r) * np.cos(pitch) * np.sin(y) - np.sin(r) * np.sin(pitch) * np.cos(y),
                        np.cos(r) * np.cos(pitch) * np.cos(y) + np.sin(r) * np.sin(pitch) * np.sin(y),
                    ]
                    p = p @ pose_matrix(np.r_[np.zeros(3), q]).T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                self.parts.append((names.index(link.get("name")), p, np.asarray(faces, dtype=np.int32)))
        if not self.parts:
            raise ValueError("No matching URDF visual meshes for robot occlusion")

    def world_mesh(self, poses):
        points, faces, offset = [], [], 0
        for body, p, f in self.parts:
            points.append(p @ pose_matrix(poses[body]).T + poses[body, :3])
            faces.append(f + offset)
            offset += len(p)
        return np.concatenate(points).astype(np.float32), np.concatenate(faces)


class SensorRig:
    def __init__(self, soil, body_names, visualizers, args):
        self.soil, self.args = soil, args
        self.scene = TriangleScene(soil.device)
        self.names = body_names
        self.robot = RobotGeometry(body_names, Path(__file__).resolve().parent.parent / "ros2/urdf")
        self.rays = fan(args.lidar_step_deg)
        self.observed = self.oracle = np.full((HEIGHTMAP_SIZE, HEIGHTMAP_SIZE), np.nan, np.float32)
        self.cabin_id = body_names.index("upper_carriage")
        self.points = np.empty((0, 3), np.float32)
        self.colors = self.points.copy()
        self.status = "Waiting for scan"
        from map_panel import ImageTextures

        self.textures = ImageTextures()
        self.scan_count = 0
        self.depth_images = {}
        self.ground = None
        # Preserve calibrated field of view at quarter resolution. These are geometric Z-depth images.
        self.camera_rays = {
            name: pinhole_rays(fx * 160 / DEPTH_SIZE[0], fy * 120 / DEPTH_SIZE[1], 160, 120)
            for name, _, _, _, fx, fy in DEPTH_CAMERAS
        }
        self.stream = self.writer = None
        if args.record_dir:
            self.directory = args.record_dir / "sensors"
            self.directory.mkdir(parents=True, exist_ok=True)
            self.stream = (self.directory / "metrics.csv").open("w", newline="")
        for visualizer in visualizers:
            if type(visualizer).__name__ == "NewtonGLVisualizer":
                visualizer.register_ui_callback(self.draw, position="free")

    def sync_soil(self, surface):
        """Upload the displayed ground and moving-soil surface to the ray scene."""
        soil = self.soil
        ground = (surface.display_height, soil.indices)
        if self.ground is None or any(new is not old for new, old in zip(ground, self.ground)):
            p, f, _ = soil.mesh(height=surface.display_height, normals=False)
            self.scene.update("ground", p, f)
            self.ground = ground
        vertices, indices, _ = (
            surface.mesh.to_arrays() if surface.mesh is not None and surface.triangles else (None, None, None)
        )
        moving_p = vertices.numpy() if vertices is not None else np.empty((0, 3), np.float32)
        moving_f = indices.numpy() if indices is not None else np.empty(0, np.int32)
        # Sparse spray has no isosurface. Match the small displayed grains with octahedral proxies.
        mask = surface.spray_shown & (soil.active != 0)
        spray = soil.positions[mask]
        if len(spray):
            offsets = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32) * (
                0.2 * soil.cfg.cell
            )
            octa = np.array([0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5], np.int32)
            spray_f = (np.arange(len(spray))[:, None] * 6 + octa + len(moving_p)).ravel()
            moving_p = np.concatenate((moving_p, (spray[:, None, :] + offsets).reshape(-1, 3)))
            moving_f = np.concatenate((moving_f, spray_f))
        self.scene.update("moving", moving_p, moving_f)

    def update(self, surface, poses, sim_time):
        begin = time.perf_counter()
        soil = self.soil
        self.sync_soil(surface)
        cabin = poses[self.cabin_id]
        self.oracle = ground_map(soil, surface.display_height, cabin)
        if self.args.lidars or self.args.depth_cameras:
            self.scan_robot(poses)
        points, labels, origins = [np.empty((0, 3), np.float32)], [np.empty(0, np.int32)], [np.empty((0, 3))]
        for name, parent, xyz, quat in LIDARS if self.args.lidars else ():
            pose = poses[self.names.index(parent)]
            rotation = pose_matrix(pose)
            origin = pose[:3] + rotation @ np.asarray(xyz)
            directions = self.rays @ pose_matrix(np.r_[np.zeros(3), quat]).T @ rotation.T
            starts = np.broadcast_to(origin, directions.shape).copy()
            distance, label = self.scene.cast(starts, directions, far=self.args.lidar_range)
            points.append(starts + distance[:, None] * directions)
            labels.append(label)
            origins.append(starts)
        points, labels, origins = np.concatenate(points), np.concatenate(labels), np.concatenate(origins)
        soil_hits = (labels == 1) | (labels == 2)
        self.observed = rasterize(points[soil_hits], cabin)
        return self.finish_scan(poses, sim_time, begin, points, labels, origins, soil_hits)

    def scan_robot(self, poses):
        """Put the robot and its support block into the ray scene as occluders."""
        soil = self.soil
        robot_p, robot_f = self.robot.world_mesh(poses)
        # The artificial support block also occludes sensors; omit it from soil maps.
        stand = np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            np.float32,
        )
        stand *= np.array([0.18, 0.15, -soil.cfg.floor / 2], np.float32)
        stand[:, 2] += soil.cfg.floor / 2
        stand_f = np.array(
            [
                0,
                2,
                1,
                0,
                3,
                2,
                4,
                5,
                6,
                4,
                6,
                7,
                0,
                1,
                5,
                0,
                5,
                4,
                1,
                2,
                6,
                1,
                6,
                5,
                2,
                3,
                7,
                2,
                7,
                6,
                3,
                0,
                4,
                3,
                4,
                7,
            ],
            np.int32,
        )
        self.scene.update("robot", np.concatenate((robot_p, stand)), np.r_[robot_f, stand_f + len(robot_p)])

    def finish_scan(self, poses, sim_time, begin, points, labels, origins, soil_hits):
        cabin = poses[self.cabin_id]
        self.points = points[labels != 0].astype(np.float32)
        palette = np.array([[0, 0, 0], [0.15, 0.9, 0.3], [1, 0.8, 0.1], [1, 0.2, 0.15]], np.float32)
        self.colors = palette[labels[labels != 0]]
        if self.args.depth_cameras:
            for name, parent, xyz, quat, _, _ in DEPTH_CAMERAS:
                pose = poses[self.names.index(parent)]
                rotation = pose_matrix(pose)
                origin = pose[:3] + rotation @ np.asarray(xyz)
                rays = self.camera_rays[name]
                optical_rotation = rotation @ pose_matrix(np.r_[np.zeros(3), quat])
                self.depth_images[name] = capture_depth(self.scene, origin, optical_rotation, rays, 160, 120)
        scan_ms = (time.perf_counter() - begin) * 1000
        row = dict(
            time_s=sim_time,
            scan_ms=scan_ms,
            rays=len(labels),
            soil_hits=int(soil_hits.sum()),
            robot_hits=int((labels == 3).sum()),
            moving_hits=int((labels == 2).sum()),
            observed_cells=int(np.isfinite(self.observed).sum()),
            oracle_cells=int(np.isfinite(self.oracle).sum()),
        )
        self.status = f"{scan_ms:.1f} ms | ground map" + (
            f" | {len(labels)} rays | soil {row['soil_hits']} | self {row['robot_hits']}" if len(labels) else ""
        )
        if self.stream:
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.stream.flush()
            np.savez_compressed(
                self.directory / f"scan_{sim_time:08.3f}.npz",
                time_s=sim_time,
                observed=self.observed,
                oracle=self.oracle,
                observed_mono8=map_pixels(self.observed),
                oracle_mono8=map_pixels(self.oracle),
                points_world=points,
                labels=labels,
                origins_world=origins,
                cabin_pose=cabin,
                body_poses=poses,
                **{f"depth_{name}_m": depth for name, depth in self.depth_images.items()},
            )
        self.scan_count += 1
        return row

    def draw(self, imgui):
        imgui.set_next_window_size(imgui.ImVec2(690, 435), imgui.Cond_.first_use_ever)
        if imgui.begin("Height maps")[0]:
            imgui.text(self.status)
            imgui.text("Cabin frame | 64x64 | +X up, +Y left | checker = unknown")
            maps = [("Virtual lidar", self.observed)] if self.args.lidars else []
            for name, height in [*maps, ("Ground (no robot, no carried sand)", self.oracle)]:
                imgui.begin_group()
                imgui.text(name)
                gray = map_pixels(height)
                yy, xx = np.indices(height.shape)
                gray = np.where(np.isfinite(height), gray, np.where((yy // 4 + xx // 4) % 2, 30, 55)).astype(np.uint8)
                self.textures.draw(imgui, name, gray, self.scan_count, (320, 320))
                imgui.end_group()
                if name == "Virtual lidar":
                    imgui.same_line()
            imgui.text("Height: -0.30 m black to 0 m white, relative to cabin. No temporal fill.")
        imgui.end()
        if self.depth_images:
            if imgui.begin("Geometric depth cameras (no RTX)")[0]:
                for name, depth in self.depth_images.items():
                    imgui.begin_group()
                    imgui.text(f"{name}: 160x120, optical Z")
                    gray = np.where(np.isfinite(depth), 255 * np.clip(1 - depth / 2, 0, 1), 0).astype(np.uint8)
                    self.textures.draw(imgui, name, gray, self.scan_count, (320, 240))
                    imgui.end_group()
                    if name == "cabin":
                        imgui.same_line()
                imgui.text("Black = missing or 2 m; white = near. Ideal geometry, no stereo noise.")
            imgui.end()

    def close(self):
        self.textures.close()
        if self.stream:
            self.stream.close()
