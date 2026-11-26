"""One adapter to the robot's projection, masking, fusion and FastDEM code.

Used by the Isaac viewer, paired depth comparisons and offline filter benchmarks.
The robot checkout remains the owner of the algorithms and default parameters.
"""

import copy
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import ThreadpoolController

ROOT = Path(__file__).resolve().parent
STREAMS = {"cabin": "cam2", "arm": "cam1"}


@functools.lru_cache(maxsize=1)
def blas_controller():
    # Enumerating Kit's loaded DLLs on every scan costs far more than mapping.
    return ThreadpoolController()


def mapping_arguments(parser):
    parser.add_argument(
        "--mapping",
        choices=("robot", "preview"),
        default="robot",
        help="Robot filters + FastDEM (default), or the legacy preview",
    )
    parser.add_argument(
        "--robot-repo",
        type=Path,
        help="kaivuriprokkis checkout; defaults to KAIVURI_REPO or ~/Documents/GitHub/kaivuriprokkis",
    )
    parser.add_argument(
        "--fastdem", type=Path, help="Compiled robot FastDEM adapter; defaults to the local portable build"
    )


def robot_checkout(path=None):
    repo = Path(path or os.environ.get("KAIVURI_REPO", Path.home() / "Documents/GitHub/kaivuriprokkis")).resolve()
    if not (repo / "native_stack/tools/heightmap_bench/mapper.py").is_file():
        raise FileNotFoundError(f"Robot mapper not found in {repo}; pass --robot-repo PATH")
    return repo


def fastdem_binary(path=None):
    executable = "fastdem_replay.exe" if os.name == "nt" else "fastdem_replay"
    build = "build-mingw" if os.name == "nt" else "build"
    result = Path(path) if path else ROOT / "logs/fastdem" / build / executable
    if not result.is_file():
        raise FileNotFoundError(f"FastDEM not found: {result}. Build it with build_fastdem.py or pass --fastdem PATH")
    return result.resolve()


@functools.lru_cache(maxsize=4)
def load_robot_mapper(repo):
    repo = robot_checkout(repo)
    tools = repo / "native_stack/tools"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("robot_heightmap_mapper", tools / "heightmap_bench/mapper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def millimetres(height):
    result = np.full(height.shape, -32768, np.int16)
    known = np.isfinite(height)
    result[known] = np.clip(np.rint(height[known] * 1000), -32767, 32767).astype(np.int16)
    return result


class RobotPolicy:
    def __init__(self, repo, dataset, session=0, *, empirical_correction=False, overrides=None):
        self.repo = robot_checkout(repo)
        self.mapper = load_robot_mapper(self.repo)
        native = self.repo / "native_stack"
        self.policy = json.loads((native / "config/heightmap_pipeline.json").read_text())
        self.policy["projection"].update(overrides or {})
        self.projection = SimpleNamespace(**self.policy["projection"], arm_model=str(native / "config/arm_model.npz"))
        self.mapper.sh.load_arm(self.projection)
        self.arm = self.mapper.sh.ARM
        self.blas = blas_controller()
        self.metadata = json.loads((dataset / "meta/heightmaps.json").read_text())
        self.calibration = copy.deepcopy(self.metadata["calibration"])
        if not empirical_correction:
            self.calibration["depth_correction"] = {}
        files = {key: dataset / f"native/session-{session:03d}/{key}.depth.json" for key in STREAMS.values()}
        self.projector = self.mapper.DepthProjector(self.calibration, files, self.projection)
        self.config = native / "tools/heightmap_bench/fastdem.yaml"
        self.revision = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()

    def provenance(self):
        sources = [
            "tools/heightmap_bench/mapper.py",
            "tools/sand_height.py",
            "config/heightmap_pipeline.json",
            "config/arm_model.npz",
            "tools/heightmap_bench/fastdem.yaml",
        ]
        return {
            "backend": "robot",
            "robot_repo": str(self.repo),
            "robot_revision": self.revision,
            "pipeline": self.policy,
            "calibration": self.calibration,
            "source_sha256": {
                path: hashlib.sha256((self.repo / "native_stack" / path).read_bytes()).hexdigest() for path in sources
            },
            "map_age_units": "accepted paired map updates; 65535 means unknown",
        }

    def set_optical_poses(self, states, poses):
        """Use actual simulated camera poses, expressed in the same frame as FK.

        Resolve both mounts before scanning either camera, including camera-body exclusion.
        """
        for name, key in STREAMS.items():
            rotation, translation = self.mapper.sh.link_pose(states[name], self.projector.mounts[key].link)
            actual = poses[name]
            mount = self.projector.mounts[key]
            replacement = self.mapper.sh.Mount(
                mount.link,
                rotation.T @ actual[:3, :3],
                rotation.T @ (actual[:3, 3] - translation),
                mount.n,
                mount.d,
            )
            self.projector.mounts[key] = replacement
            self.projector.all_mounts[key] = replacement

    def scan(self, name, image, state):
        # The upstream module uses one global arm model; keep independent sweep profiles isolated.
        self.mapper.sh.ARM = self.arm
        with self.blas.limit(limits=1, user_api="blas"):
            return self.projector.scan(STREAMS[name], image.astype("<u2", copy=False).tobytes(), state)

    def fuse(self, scans, yaw, engine):
        with self.blas.limit(limits=1, user_api="blas"):
            return self.mapper.update(
                engine,
                [scans[name] for name in STREAMS],
                yaw,
                self.projection.arm_min_view_deg,
                self.projection.arm_min_forward_m,
            )

    def engine(self, path=None):
        return self.mapper.FastDEM(fastdem_binary(path), self.config)


class UsdRobotPose:
    """Translate live USD poses to the robot's level frame and FK status contract."""

    def __init__(self, stage, repo, ground_z):
        import yaml

        self.stage, self.ground_z = stage, ground_z
        config = robot_checkout(repo) / "configuration_files/profiles/jetson/control_config.yaml"
        self.tip = np.asarray(yaml.safe_load(config.read_text())["robot"]["tool"]["parent_to_tip_xyz"])

    def sample(self, camera_poses):
        from scipy.spatial.transform import Rotation

        from pxr import UsdGeom

        root = "/World/excavator_bucket"
        cache = UsdGeom.XformCache()
        world = {
            name: np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(f"{root}/{name}"))).T
            for name in ("lower_carriage", "upper_carriage", "lift_boom", "tilt_boom", "tool_body")
        }
        base = world.pop("lower_carriage")
        base_quaternion = Rotation.from_matrix(base[:3, :3]).as_quat(scalar_first=True)
        yaw = 2 * np.arctan2(base_quaternion[3], base_quaternion[0])
        c, s = np.cos(yaw), np.sin(yaw)
        stage_to_native = np.eye(4)
        stage_to_native[:3, :3] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
        stage_to_native[:3, 3] = -stage_to_native[:3, :3] @ [base[0, 3], base[1, 3], self.ground_z]
        links = np.stack([stage_to_native @ matrix for matrix in world.values()])
        quaternions = Rotation.from_matrix(links[:, :3, :3]).as_quat(scalar_first=True)
        # Match the controller's swing/twist extraction; Euler heading includes a tilt-dependent yaw error.
        slew = 2 * np.arctan2(quaternions[0, 3], quaternions[0, 0])
        state = {
            "fk_valid": True,
            "imu_healthy": True,
            "fk_cabin_tilt": True,
            "fk_joint_xyz_m": links[:, :3, 3],
            "fk_link_quat_wxyz": quaternions,
            "fk_xyz_m": links[3, :3, :3] @ self.tip + links[3, :3, 3],
            "slew_yaw_rad": float((slew + np.pi) % (2 * np.pi) - np.pi),
        }
        return state, {name: stage_to_native @ matrix for name, matrix in camera_poses.items()}, base, stage_to_native
