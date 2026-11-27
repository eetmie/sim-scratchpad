"""D435i optical depth and dataset-shaped terrain maps for the registered USD scene.

Run ``python depth_mapping.py --help`` for a single capture, or use the live
``launch_3dgs.py --demo`` viewer. All USD edits are transient.
"""

import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
from heightmap_config import load_profile as load_profile
from robot_mapping import STREAMS, RobotPolicy, UsdRobotPose, fastdem_binary, millimetres

ROOT = Path(__file__).resolve().parent
UNKNOWN_MM = np.int16(-32768)
UNKNOWN_AGE = np.uint16(65535)
logger = logging.getLogger(__name__)


def optical_pose(ir_pose: dict, sensor: dict) -> np.ndarray:
    """Depth optical -> parent, as a column-vector matrix in metres.

    RealSense extrinsics map depth points into IR coordinates. Compose them
    with the fitted IR mount, including the arm camera's stereo baseline.
    """
    from scipy.spatial.transform import Rotation

    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(ir_pose["quaternion_xyzw"]).as_matrix()
    pose[:3, 3] = ir_pose["translation_m"]
    extr = sensor["depth_to_ir_extrinsics"]
    depth_to_ir = np.eye(4)
    depth_to_ir[:3, :3] = np.asarray(extr["rotation_column_major"]).reshape(3, 3, order="F")
    depth_to_ir[:3, 3] = extr["translation_m"]
    return pose @ depth_to_ir


def deproject(depth_m: np.ndarray, sensor: dict, pose: np.ndarray, stride: int = 1) -> np.ndarray:
    """Recorded-orientation optical Z [m] -> world XYZ [m], invalid pixels omitted.

    The matrix maps the unrotated right/down/forward sensor frame into world.
    Range is radial, matching the robot mapper's 0.1--0.9 m working volume.
    ``stride`` subsamples sensor pixels before projection, retaining their coordinates.
    """
    depth = depth_m[::-1, ::-1] if sensor["rotation_deg"] == 180 else depth_m
    depth = depth[::stride, ::stride]
    intr = sensor["depth_intrinsics"]
    v, u = np.indices(depth.shape)
    z = depth.ravel()
    points = np.column_stack(
        (
            (u.ravel() * stride - intr["ppx"]) * z / intr["fx"],
            (v.ravel() * stride - intr["ppy"]) * z / intr["fy"],
            z,
        )
    )
    distance = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & (z > 0) & (distance >= 0.1) & (distance <= 0.9)
    return points[valid] @ pose[:3, :3].T + pose[:3, 3]


def rasterize(points: np.ndarray, x_m: tuple, y_m: tuple, cell_m: float) -> np.ndarray:
    """Median height [mm] per observed cell; row 0 is +X and column 0 is +Y."""
    rows, cols = (round((hi - lo) / cell_m) for lo, hi in (x_m, y_m))
    out = np.full((rows, cols), UNKNOWN_MM, dtype=np.int16)
    valid = np.isfinite(points).all(axis=1)
    valid &= (points[:, 0] >= x_m[0]) & (points[:, 0] < x_m[1])
    valid &= (points[:, 1] >= y_m[0]) & (points[:, 1] < y_m[1])
    p = points[valid]
    if not len(p):
        return out
    r = rows - 1 - np.floor((p[:, 0] - x_m[0]) / cell_m).astype(int)
    c = cols - 1 - np.floor((p[:, 1] - y_m[0]) / cell_m).astype(int)
    cells = r * cols + c
    order = np.lexsort((p[:, 2], cells))
    ids, starts, counts = np.unique(cells[order], return_index=True, return_counts=True)
    z = p[order, 2]
    median = (z[starts + (counts - 1) // 2] + z[starts + counts // 2]) / 2
    out.ravel()[ids] = np.clip(np.rint(median * 1000), -32767, 32767).astype(np.int16)
    return out


def map_image(mm: np.ndarray, limit_m: float = 0.12) -> np.ndarray:
    """Dataset grayscale convention: ground 128, +/-120 mm 16/240, unknown 0."""
    image = np.rint(128 + 112 * np.clip(mm.astype(float) / (limit_m * 1000), -1, 1)).astype(np.uint8)
    image[mm == UNKNOWN_MM] = 0
    return image


def level_points(points: np.ndarray, origin: np.ndarray, yaw: float, ground_z: float) -> np.ndarray:
    """World points -> level frame; heights are relative to the registered sand surface."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = (points - [origin[0], origin[1], ground_z]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local[(local[:, 2] >= -0.35) & (local[:, 2] <= 0.3)]


def level_pose(origin: np.ndarray, yaw: float, ground_z: float) -> np.ndarray:
    """Ground-relative level map frame -> stage/world, in metres."""
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[:3, 3] = [origin[0], origin[1], ground_z]
    return pose


class TerrainMapper:
    """Latest front crop and a persistent surrounding map for a stationary base.

    Each observed cell receives the current scan's median. Unknown cells stay
    unknown until seen; no inpainting. This is a lightweight mapper, not FastDEM.
    Ages count nominal dataset frames (30 Hz), rather than render frames.
    """

    def __init__(self, profile: dict):
        self.profile = profile
        cfg = profile["surround"]
        size = round(np.ptp(cfg["xy_m"]) / cfg["cell_m"])
        self.surround = np.full((size, size), UNKNOWN_MM, np.int16)
        self.seen_at = np.full((size, size), -np.inf)

    def update(self, front_points: np.ndarray, base_points: np.ndarray, stamp_s: float) -> dict:
        front = self.profile["front"]
        front_mm = rasterize(front_points, front["x_m"], front["y_m"], front["cell_m"])
        cfg = self.profile["surround"]
        radius = np.linalg.norm(base_points[:, :2], axis=1)
        ring = base_points[(radius >= cfg["radius_m"][0]) & (radius <= cfg["radius_m"][1])]
        current = rasterize(ring, cfg["xy_m"], cfg["xy_m"], cfg["cell_m"])
        seen = current != UNKNOWN_MM
        self.surround[seen] = current[seen]
        self.seen_at[seen] = stamp_s
        age = np.full(current.shape, UNKNOWN_AGE, np.uint16)
        known = np.isfinite(self.seen_at)
        age[known] = np.clip(np.rint((stamp_s - self.seen_at[known]) * self.profile["fps"]), 0, 65534)
        return {"front_mm": front_mm, "surround_mm": self.surround.copy(), "surround_age_frames": age}


class DepthMapping:
    """Render depth and use the robot's masking, fusion and FastDEM in the viewer.

    Requires a running SimulationApp and an open stage with the session edit target.
    Captures preserve timeline playback and use live USD transforms, so sensors
    follow their parent links. The scene starts with physics paused in the viewer.
    """

    def __init__(
        self,
        stage,
        dataset: Path | None,
        out: Path,
        ground_z: float | None = None,
        panel: bool = True,
        session: int = 0,
        *,
        robot_repo: Path | None = None,
        fastdem: Path | None = None,
        mapping: str = "robot",
        cad_reference: bool = False,
        config: Path | None = None,
    ):
        from camera_calibration.usd_cameras import author, cameras, describe
        from scipy.spatial.transform import Rotation

        import omni.replicator.core as rep
        from pxr import UsdGeom

        self.stage = stage
        self.out = out
        self.profile = load_profile(dataset, session, config=config)
        ground_z = self.profile["ground_z_stage_m"] if ground_z is None else ground_z
        self.ground_z = ground_z
        self.backend = mapping
        self.cad_reference = cad_reference or mapping == "preview"
        self.robot_engine = None
        if mapping == "robot":
            self.robot_policy = RobotPolicy(robot_repo, dataset, session, config=config)
            self.pose_bridge = UsdRobotPose(stage, self.robot_policy.repo, ground_z, tip=self.robot_policy.tool_tip)
            self.fastdem = fastdem_binary(fastdem)
            self.profile.update(self.robot_policy.provenance())
            self.profile["map_hz"] = self.robot_policy.policy["runtime"]["map_hz"]
            self.mapper = None
        else:
            self.robot_policy = None
            self.mapper = TerrainMapper(self.profile)
        self.products = {}
        self.last = None
        self.started = time.monotonic()
        self.status = None
        self.window = None
        self.providers = {}
        self.image_buffers = {}
        self.latest = {}
        self.sample_count = 0
        self.base_anchor = None
        self.warmed_up = False
        self.timings = deque(maxlen=512)
        self.last_completed = None
        out.mkdir(parents=True, exist_ok=True)
        rep.orchestrator.set_capture_on_play(False)
        for name, camera in cameras(stage).items():
            sensor = self.profile["sensors"][name]
            info = describe(camera)
            pose = optical_pose(info["mount"], sensor)
            path = str(camera.GetPrim().GetParent().GetPath()) + f"/depth_{name}"
            depth_camera = UsdGeom.Camera.Define(stage, path)
            depth_camera.CreateClippingRangeAttr().Set((0.05, 10.0))
            intr = sensor["depth_intrinsics"]
            info.update(
                prim=path,
                mount={
                    "translation_m": pose[:3, 3].tolist(),
                    "quaternion_xyzw": Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
                },
                intrinsics={
                    **{k: intr[k] for k in ("width", "height", "fx", "fy", "coeffs")},
                    "cx": intr["ppx"],
                    "cy": intr["ppy"],
                    "model": "none",
                },
                image_rotation_deg=sensor["rotation_deg"],
            )
            author(depth_camera, info)
            product = rep.create.render_product(path, (intr["width"], intr["height"]), force_new=True)
            depth = rep.annotators.get("distance_to_image_plane", device="cpu")
            instances = rep.annotators.get("instance_id_segmentation", device="cpu") if self.cad_reference else None
            depth.attach(product.path)
            if instances is not None:
                instances.attach(product.path)
            self.products[name] = (depth_camera, product, depth, instances)
        self.profile.update(
            ground_z_stage_m=ground_z,
            depth_model="ideal RTX optical Z; no RealSense disparity noise or empirical depth correction",
            fusion="robot cabin-first fusion and FastDEM"
            if mapping == "robot"
            else "latest observation replaces surrounding cells",
            axes="row decreases X, column decreases Y",
            unknown_mm=int(UNKNOWN_MM),
            unknown_age_frames=int(UNKNOWN_AGE),
            mapping_pixel_stride=self.robot_policy.projection.stride if self.robot_policy else 2,
            mapping_backend=mapping,
            cad_reference=self.cad_reference,
            point_cloud="self-filtered camera candidates before arm-view and cabin-priority fusion",
        )
        if mapping == "robot":
            self.profile["fastdem_executable"] = str(self.fastdem)
            self.profile["surround"]["age_units"] = "accepted paired map updates"
        (out / "profile.json").write_text(json.dumps(self.profile, indent=2) + "\n")
        self.metrics_stream = (out / "updates.jsonl").open("w", buffering=1)
        if panel:
            self._create_panel()

    def _create_panel(self) -> None:
        import omni.ui as ui

        self.window = ui.Window("D435i depth and terrain maps", width=690, height=700)
        with self.window.frame, ui.VStack(spacing=6):
            ui.Label(f"D435i | {self.backend} mapper | red = robot mask | black = unknown", height=24)
            with ui.HStack(height=230):
                for name in STREAMS:
                    with ui.VStack():
                        ui.Label(f"{name} optical depth (0.1-0.9 m)", height=22)
                        provider = ui.ByteImageProvider()
                        self.providers[name] = provider
                        ui.ImageWithProvider(provider)
            with ui.HStack(height=300):
                for name, label in (("front", "Front 64x64 / cabin yaw"), ("surround", "Surround 160x160 / base")):
                    with ui.VStack():
                        ui.Label(label, height=22)
                        provider = ui.ByteImageProvider()
                        self.providers[name] = provider
                        ui.ImageWithProvider(provider)
            self.status = ui.Label("Waiting for depth...", height=36, word_wrap=True)
            with ui.HStack(height=26):
                ui.Button("Save current depth + maps", clicked_fn=self.save)
                ui.Button("Clear accumulated map", clicked_fn=self.clear)

    def clear(self) -> None:
        if self.robot_policy is not None:
            if self.robot_engine is not None:
                self.robot_engine.__exit__(None, None, None)
                self.robot_engine = None
        else:
            self.mapper = TerrainMapper(self.profile)

    def capture(self) -> dict:
        """Render both cameras at the current pose, without throttling or map fusion.

        Results contain recorded-orientation ``z16`` and ``robot_mask`` images
        and an unrotated optical-to-world column-vector ``pose`` in metres.
        """
        from camera_calibration.usd_cameras import ROBOT, describe
        from scipy.spatial.transform import Rotation

        import omni.replicator.core as rep

        now = time.monotonic()
        self.capture_clock = {
            "capture_host_monotonic_ns": np.array(time.monotonic_ns(), dtype=np.int64),
            "capture_host_unix_ns": np.array(time.time_ns(), dtype=np.int64),
        }
        render_ms = 0.0
        # Kit can schedule the first render before newly attached host buffers exist.
        # Wait for both depth and instance buffers; keep a bound on failed renderer startup.
        deadline = now + 30
        buffers = {}
        while len(buffers) != len(self.products):
            render_start = time.perf_counter()
            rep.orchestrator.step(rt_subframes=4 if not self.warmed_up else 1, delta_time=0.0, pause_timeline=False)
            render_ms += (time.perf_counter() - render_start) * 1000
            buffers = {}
            for name, (_, _, depth, instances) in self.products.items():
                intr = self.profile["sensors"][name]["depth_intrinsics"]
                image = np.asarray(depth.get_data()).copy()
                labels = instances.get_data() if instances is not None else None
                expected = (intr["height"], intr["width"])
                if image.shape == expected and (
                    instances is None
                    or (
                        isinstance(labels, dict)
                        and np.shape(labels.get("data")) == expected
                        and "idToLabels" in labels.get("info", {})
                    )
                ):
                    buffers[name] = (image, labels)
            if len(buffers) != len(self.products) and time.monotonic() >= deadline:
                raise TimeoutError("D435i depth and instance buffers were not ready within 30 seconds")
        captures = {}
        for name, (camera, _, _, _) in self.products.items():
            sensor = self.profile["sensors"][name]
            image, labels = buffers[name]
            robot_ids = (
                [int(k) for k, path in labels["info"]["idToLabels"].items() if str(path).startswith(ROBOT + "/")]
                if labels is not None
                else []
            )
            self_mask = np.isin(labels["data"], robot_ids) if labels is not None else None
            image[~np.isfinite(image) | (image <= 0) | (image >= sensor["depth_scale_m"] * 65536)] = 0
            if sensor["rotation_deg"] == 180:
                image = image[::-1, ::-1].copy()
            # Quantize at the recorder's 0.1 mm Z16 scale before deprojection.
            z16 = np.clip(np.rint(image / sensor["depth_scale_m"]), 0, 65535).astype(np.uint16)
            info = describe(camera)["optical_to_world"]
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(info["quaternion_xyzw"]).as_matrix()
            pose[:3, 3] = info["translation_m"]
            if self_mask is not None and sensor["rotation_deg"] == 180:
                self_mask = self_mask[::-1, ::-1]
            captures[name] = {"z16": z16, "pose": pose}
            if self_mask is not None:
                captures[name]["robot_mask"] = self_mask.copy()
        self.warmed_up = True
        self.capture_timings = {"render_ms": render_ms, "readback_ms": (time.monotonic() - now) * 1000 - render_ms}
        return captures

    def update(self, force: bool = False) -> bool:
        """Capture at 5 Hz wall time; return whether a new map was produced."""
        from camera_calibration.usd_cameras import ROBOT

        from pxr import UsdGeom

        now = time.monotonic()
        if not force and self.last is not None and now - self.last < 1 / self.profile["map_hz"]:
            return False
        captures = self.capture()
        if self.robot_policy is not None:
            return self._update_robot(captures, now)
        cache = UsdGeom.XformCache()
        cloud = []
        arrays = {}
        diagnostics = {}
        for name, capture in captures.items():
            sensor = self.profile["sensors"][name]
            z16, self_mask, pose = (capture[k] for k in ("z16", "robot_mask", "pose"))
            image = z16.astype(np.float32) * sensor["depth_scale_m"]
            filtered = image.copy()
            filtered[self_mask] = 0
            points = deproject(filtered, sensor, pose, stride=self.profile["mapping_pixel_stride"])
            cloud.append(points)
            arrays[f"depth_{name}_z16"] = z16
            arrays[f"optical_to_world_{name}"] = pose
            diagnostics[name] = {"points_in_range": len(points), "self_hit_pixels": int(self_mask.sum())}
            if self.providers:
                gray = np.rint(255 * np.clip((image - 0.1) / 0.8, 0, 1)).astype(np.uint8)
                gray[(image <= 0) | (image > 0.9)] = 0
                self._image(name, gray[::2, ::2])
        points = np.concatenate(cloud)
        cabin = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(f"{ROBOT}/upper_carriage"))).T
        base = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(f"{ROBOT}/lower_carriage"))).T
        if self.base_anchor is not None and not np.allclose(base, self.base_anchor, atol=1e-5):
            self.clear()
        self.base_anchor = base.copy()
        front_points = level_points(points, cabin[:3, 3], np.arctan2(cabin[1, 0], cabin[0, 0]), self.ground_z)
        base_points = level_points(points, base[:3, 3], np.arctan2(base[1, 0], base[0, 0]), self.ground_z)
        stamp = time.monotonic() - self.started
        arrays.update(self.mapper.update(front_points, base_points, stamp))
        arrays.update(points_world_m=points.astype(np.float32), stamp_s=np.array(stamp))
        arrays.update(self.capture_clock)
        arrays.update(
            front_to_world=level_pose(cabin[:3, 3], np.arctan2(cabin[1, 0], cabin[0, 0]), self.ground_z),
            surround_to_world=level_pose(base[:3, 3], np.arctan2(base[1, 0], base[0, 0]), self.ground_z),
        )
        self.latest = arrays
        self.diagnostics = diagnostics
        self.sample_count += 1
        self.last = now
        self.capture_ms = (time.monotonic() - now) * 1000
        self.diagnostics["capture_ms"] = self.capture_ms
        if self.providers:
            for name in ("front", "surround"):
                self._image(name, map_image(arrays[f"{name}_mm"], self.profile["height_color_limit_m"]))
            coverage = np.mean(arrays["front_mm"] != UNKNOWN_MM) * 100
            self.status.text = (
                f"Front observed {coverage:.1f}% | {len(points):,} points | capture {self.capture_ms:.0f} ms"
            )
        return True

    def _update_robot(self, captures, started):
        # Include export/preview point transforms, not just the filter calls.
        with self.robot_policy.blas.limit(limits=1, user_api="blas"):
            return self._update_robot_limited(captures, started)

    def _update_robot_limited(self, captures, started):
        """The same RobotPolicy used for recorded real/sim mapping; no CAD-ID filtering."""
        policy = self.robot_policy
        projected_at = time.perf_counter()
        state, poses, base, stage_to_native = self.pose_bridge.sample(
            {name: cap["pose"] for name, cap in captures.items()}
        )
        if self.base_anchor is not None and not np.allclose(base, self.base_anchor, atol=1e-5):
            self.clear()
        self.base_anchor = base.copy()
        policy.set_optical_poses({name: state for name in STREAMS}, poses)
        scans = {name: policy.scan(name, cap["z16"], state) for name, cap in captures.items()}
        projection_ms = (time.perf_counter() - projected_at) * 1000
        if self.robot_engine is None:
            self.robot_engine = policy.engine(self.fastdem).__enter__()
        fused_at = time.perf_counter()
        front, surround, age = policy.fuse(scans, state["slew_yaw_rad"], self.robot_engine)
        fusion_ms = (time.perf_counter() - fused_at) * 1000
        arrays = {
            "front_mm": millimetres(front),
            "surround_mm": millimetres(surround),
            "surround_age_updates": age.copy(),
            "surround_age_frames": age.copy(),
        }
        export_started = time.perf_counter()
        native_to_stage = np.linalg.inv(stage_to_native)
        arrays.update(self.capture_clock)
        arrays.update(
            front_to_world=native_to_stage @ level_pose(np.zeros(3), state["slew_yaw_rad"], 0),
            surround_to_world=native_to_stage.copy(),
        )
        cloud, diagnostics = [], {}
        depth_preview_ms = 0.0
        for name, capture in captures.items():
            scan = scans[name]
            mask = scan[4]
            points = scan[1] - [0, 0, policy.projector.mounts[STREAMS[name]].d]
            cloud.append(points @ native_to_stage[:3, :3].T + native_to_stage[:3, 3])
            arrays[f"depth_{name}_z16"] = capture["z16"]
            arrays[f"optical_to_world_{name}"] = capture["pose"]
            arrays[f"self_mask_{name}"] = mask
            diagnostics[name] = {
                "points_in_range": len(scan[0]),
                "self_rejected_pct": float(100 * np.count_nonzero(mask == 2) / max(1, np.count_nonzero(mask))),
            }
            if self.providers:
                preview_started = time.perf_counter()
                gray = np.rint(255 * np.clip((0.9 - scan[3]) / 0.8, 0, 1)).astype(np.uint8)
                gray[scan[3] <= 0] = 0
                rgb = np.repeat(gray[..., None], 3, axis=2)
                rgb[mask == 2] = (255, 55, 40)
                self._image(name, rgb)
                depth_preview_ms += (time.perf_counter() - preview_started) * 1000
        completed = time.monotonic()
        self.capture_ms = (completed - started) * 1000
        period_ms = (completed - self.last_completed) * 1000 if self.last_completed is not None else None
        timing = {
            **self.capture_timings,
            "projection_ms": projection_ms,
            "fusion_ms": fusion_ms,
            "postprocess_ms": max(0, self.capture_ms - sum(self.capture_timings.values()) - projection_ms - fusion_ms),
            "depth_preview_ms": depth_preview_ms,
            "cloud_export_ms": (time.perf_counter() - export_started) * 1000 - depth_preview_ms,
            "total_ms": self.capture_ms,
            "period_ms": period_ms,
        }
        diagnostics.update(
            timing,
            target_hz=self.profile["map_hz"],
            capture_ms=self.capture_ms,
            map_age_units="accepted paired map updates",
        )
        arrays.update(
            points_world_m=np.concatenate(cloud).astype(np.float32), stamp_s=np.array(completed - self.started)
        )
        self.latest, self.diagnostics = arrays, diagnostics
        self.timings.append(timing)
        self.sample_count += 1
        self.last, self.last_completed = started, completed
        self.metrics_stream.write(
            json.dumps({"sample": self.sample_count, "stamp_s": completed - self.started, **diagnostics}) + "\n"
        )
        if self.providers:
            for name in ("front", "surround"):
                self._image(name, map_image(arrays[f"{name}_mm"], self.profile["height_color_limit_m"]))
            self.status.text = (
                f"Robot filters + FastDEM | front {np.isfinite(front).mean() * 100:.1f}% | "
                f"render {timing['render_ms']:.0f} / filter {projection_ms:.0f} / map {fusion_ms:.0f} ms"
            )
        return True

    def _image(self, name: str, gray: np.ndarray) -> None:
        rgba = (
            np.repeat(gray[:, :, None], 4, axis=2)
            if gray.ndim == 2
            else np.concatenate((gray, np.full((*gray.shape[:2], 1), 255, np.uint8)), axis=2)
        )
        rgba[:, :, 3] = 255
        # Pass the contiguous array buffer rather than converting every byte through Python bindings.
        self.image_buffers[name] = rgba
        self.providers[name].set_data_array(rgba.ravel(), [gray.shape[1], gray.shape[0]])

    def save(self) -> None:
        """Save the current scan without resampling or touching the real recording."""
        import cv2

        if not self.latest:
            return
        np.savez_compressed(self.out / "latest.npz", **self.latest)
        for name in STREAMS:
            if not cv2.imwrite(str(self.out / f"depth_{name}_z16.png"), self.latest[f"depth_{name}_z16"]):
                raise OSError("Could not save depth PNG")
        for name in ("front", "surround"):
            image = map_image(self.latest[f"{name}_mm"], self.profile["height_color_limit_m"])
            if not cv2.imwrite(str(self.out / f"{name}.png"), image):
                raise OSError(f"Could not save {name} map PNG")
        (self.out / "diagnostics.json").write_text(json.dumps(self.diagnostics, indent=2) + "\n")
        if len(self.timings) > 1:
            samples = list(self.timings)[1:]
            report = {
                key: {
                    "median_ms": float(np.median([s[key] for s in samples])),
                    "p95_ms": float(np.percentile([s[key] for s in samples], 95)),
                }
                for key in samples[0]
                if samples[0][key] is not None
            }
            report.update(
                samples=len(samples),
                target_hz=self.profile["map_hz"],
                achieved_hz=1000 / float(np.mean([s["period_ms"] for s in samples])),
            )
            (self.out / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Saved D435i depth, world points and terrain maps to %s", self.out)

    def close(self) -> None:
        if self.robot_engine is not None:
            self.robot_engine.__exit__(None, None, None)
            self.robot_engine = None
        self.metrics_stream.close()
        for _, product, depth, instances in self.products.values():
            depth.detach(product.path)
            if instances is not None:
                instances.detach(product.path)
            product.destroy()
        if self.window:
            self.window.destroy()


def main() -> None:
    from heightmap_config import configuration_arguments
    from robot_mapping import mapping_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    configuration_arguments(parser)
    parser.add_argument("--out", type=Path, default=ROOT / "logs/depth_mapping")
    parser.add_argument("--ground-z", type=float, help="Override the profile's ground reference in stage metres")
    mapping_arguments(parser)
    args = parser.parse_args()
    # Reuse the viewer's launch and shared shader-cache settings.
    import launch_3dgs

    options = [
        "--headless",
        "--frames",
        "1",
        "--depth-maps",
        "--depth-out",
        str(args.out),
        "--mapping",
        args.mapping,
    ]
    if args.dataset is not None:
        options += ["--dataset", str(args.dataset)]
    if args.config is not None:
        options += ["--config", str(args.config)]
    if args.ground_z is not None:
        options += ["--ground-z", str(args.ground_z)]
    if args.robot_repo:
        options += ["--robot-repo", str(args.robot_repo)]
    if args.fastdem:
        options += ["--fastdem", str(args.fastdem)]
    launch_3dgs.main(options)


if __name__ == "__main__":
    main()
