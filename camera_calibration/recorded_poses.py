"""Build a paused USD replay plan from the heightmap test dataset's IR and FK timestamps."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scene_motion import apply_transforms as apply_transforms
from scipy.spatial.transform import Rotation, Slerp

from pxr import Usd, UsdGeom, UsdPhysics

from . import BASE_STAGE
from .usd_cameras import ROBOT

LINKS = ("upper_carriage", "lift_boom", "tilt_boom", "tool_body")
STREAMS = {"cabin": "clock.cam2.host_monotonic_ns", "arm": "clock.cam1.host_monotonic_ns"}
# Match the room registration's native ground origin to the authored 80 mm slew pivot.
NATIVE_TO_STAGE = np.array([0.0, 0.0, -0.0245])


def link_poses(samples, stamps):
    times = np.array(sorted(samples), dtype=np.int64)
    rows = [samples[time] for time in times]
    sensor_times = (times - times[0]) * 1e-9
    frame_times = np.clip((np.asarray(stamps) - times[0]) * 1e-9, sensor_times[0], sensor_times[-1])
    positions = np.asarray([row["fk_joint_xyz_m"] for row in rows]).reshape(-1, 12)
    positions = np.column_stack([np.interp(frame_times, sensor_times, p) for p in positions.T]).reshape(-1, 4, 3)
    quaternions = np.asarray([row["fk_link_quat_wxyz"] for row in rows]).transpose(1, 0, 2)
    rotations = np.stack(
        [Slerp(sensor_times, Rotation.from_quat(q, scalar_first=True))(frame_times).as_matrix() for q in quaternions],
        axis=1,
    )
    return rotations, positions


def replay_plan(dataset, episode, start, stop, stride, timestamp_source="ir"):
    if timestamp_source not in ("ir", "depth"):
        raise ValueError("timestamp_source must be ir or depth")
    columns = ["index", "frame_index"]
    columns += list(STREAMS.values()) if timestamp_source == "ir" else ["clock.depth_host_ns"]
    # shortcut: this export stores all observation rows in one shard, extend the reader for multi-shard takes.
    rows = pq.read_table(
        dataset / "data/chunk-000/file-000.parquet",
        columns=columns,
        filters=[("episode_index", "=", episode), ("frame_index", ">=", start), ("frame_index", "<", stop)],
    ).to_pydict()
    if rows["frame_index"] != list(range(start, stop)):
        raise ValueError("Replay range must consist of consecutive recorded frames in one episode")
    samples = {}
    with (dataset / f"native/episode-{episode:06d}/samples.jsonl").open() as source:
        for line in source:
            row = json.loads(line)["status"]
            if row["fk_valid"] and row["imu_healthy"]:
                samples[row["imu_host_monotonic_ns"]] = row
    if len(samples) < 2:
        raise ValueError("Replay needs at least two valid FK samples")

    stage = Usd.Stage.Open(str(BASE_STAGE))
    cache = UsdGeom.XformCache()
    robot_inv = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT))))
    bucket = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{ROBOT}/bucket"))
    tool = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{ROBOT}/tool_body"))
    bucket_in_tool = np.asarray(bucket * tool.GetInverse())
    lift = UsdPhysics.Joint.Get(stage, f"{ROBOT}/Joints/revolute_lift")
    q = lift.GetLocalRot1Attr().Get()
    lift_mesh_rotation = Rotation.from_quat([q.GetReal(), *q.GetImaginary()], scalar_first=True).inv().as_matrix()

    views = {}
    sample_times = np.array(sorted(samples), dtype=np.int64)
    angles = np.unwrap(np.array([samples[t]["angles_rad"] for t in sample_times]), axis=0)
    for sensor_index, (name, column) in enumerate(STREAMS.items()):
        stamps = (
            np.array(rows[column][::stride], dtype=np.int64)
            if timestamp_source == "ir"
            else np.array(rows["clock.depth_host_ns"][::stride], dtype=np.int64)[:, sensor_index]
        )
        if np.any(stamps <= 0):
            raise ValueError(f"{name}: missing {timestamp_source} exposure timestamp")
        rotations, positions = link_poses(samples, stamps)
        # The real boom FK frame points between its pins. The CAD link frame includes the joint's child rotation.
        rotations[:, 1] = rotations[:, 1] @ lift_mesh_rotation
        frames = []
        for index, frame_index, stamp, r, p in zip(
            rows["index"][::stride], rows["frame_index"][::stride], stamps, rotations, positions, strict=True
        ):
            bodies = np.repeat(np.eye(4)[None, :, :], len(LINKS), axis=0)
            bodies[:, :3, :3] = r.transpose(0, 2, 1)
            bodies[:, 3, :3] = p + NATIVE_TO_STAGE
            local = bodies @ robot_inv
            transforms = {f"{ROBOT}/{link}": matrix.tolist() for link, matrix in zip(LINKS, local, strict=True)}
            transforms[f"{ROBOT}/bucket"] = (bucket_in_tool @ local[-1]).tolist()
            frames.append(
                {
                    "index": index,
                    "frame_index": frame_index,
                    "timestamp_ns": int(stamp),
                    "fk_clamped": bool(stamp < sample_times[0] or stamp > sample_times[-1]),
                    "angles_rad": [
                        float(np.interp(stamp - sample_times[0], sample_times - sample_times[0], q)) for q in angles.T
                    ],
                    "transforms": transforms,
                }
            )
        views[name] = frames
    return {"views": views, "episode": episode, "timestamp_source": timestamp_source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, required=True, help="Exclusive frame index")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--timestamps", choices=("ir", "depth"), default="ir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.episode < 0 or not 0 <= args.start < args.stop or args.stride < 1:
        parser.error("Expected a nonnegative episode and frame range and a positive stride")
    plan = replay_plan(args.dataset, args.episode, args.start, args.stop, args.stride, args.timestamps)
    with args.out.open("x") as output:
        json.dump(plan, output, separators=(",", ":"))


if __name__ == "__main__":
    main()
