"""Compare filter coverage and flat-ground temporal stability on the native export.

Uses the same RobotPolicy and compiled FastDEM as the Isaac viewer. Profiles
are private to this process; the real robot's configuration is never changed.
"""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from replay_depth import ErrorStats, selected_depth_frames
from robot_mapping import STREAMS, RobotPolicy, fastdem_binary, robot_checkout

PROFILES = {
    "current_wide": {},
    "extra_wide": {"arm_margins_m": [0.12, 0.06, 0.12], "arm_silhouette_cells": 3},
    "smaller_margins": {"arm_margins_m": [0.06, 0.035, 0.08]},
    "wide_less_silhouette": {"arm_silhouette_cells": 1},
}


class ObservedEngine:
    def __init__(self, engine=None):
        self.engine = engine
        self.inputs = ()

    def update(self, *inputs):
        self.inputs = inputs
        return self.engine.update(*inputs) if self.engine else (None, None)


class Moments:
    def __init__(self, shape):
        self.count = np.zeros(shape, np.int32)
        self.total = np.zeros(shape)
        self.square = np.zeros(shape)

    def add(self, values, use):
        good = use & np.isfinite(values)
        self.count[good] += 1
        self.total[good] += values[good]
        self.square[good] += values[good] ** 2

    def sigma(self):
        mean = self.total / np.maximum(self.count, 1)
        return 1000 * np.sqrt(np.maximum(0, self.square / np.maximum(self.count, 1) - mean**2))


def pose_series(dataset, episode):
    samples = {}
    with (dataset / f"native/episode-{episode:06d}/samples.jsonl").open() as stream:
        for line in stream:
            state = json.loads(line)["status"]
            if state["fk_valid"] and state["imu_healthy"]:
                samples[state["imu_host_monotonic_ns"]] = state
    return np.array(sorted(samples), dtype=np.int64), [samples[t] for t in sorted(samples)]


def draw_report(out, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    series = (
        ("coverage_pct", "Current coverage in fixed sand ROI", "%"),
        ("common_temporal_sigma_p50_mm", "Temporal spread on identical cells/times", "mm (lower = steadier)"),
        ("compute_p50_ms", "Projection + FastDEM (rendering excluded)", "ms"),
    )
    for ax, (key, title, unit) in zip(axes, series):
        ax.bar(np.arange(len(names)), [results[name][key] for name in names])
        ax.set(xticks=np.arange(len(names)), xticklabels=names, title=title, ylabel=unit)
        ax.tick_params(axis="x", labelrotation=25)
    fig.savefig(out / "filter-tradeoffs.png", dpi=150)
    plt.close(fig)


def assess_sim(args, report):
    plan = json.loads((args.sim_replay / "replay.json").read_text())
    times, samples = pose_series(args.dataset, plan["episode"])
    source = json.loads((args.dataset / f"native/episode-{plan['episode']:06d}/source.json").read_text())
    results = {}
    for profile, overrides in PROFILES.items():
        policy = RobotPolicy(args.robot_repo, args.dataset, source["session"], overrides=overrides)
        total = leaked = ground = ground_rejected = 0
        for number, cabin_frame in enumerate(plan["views"]["cabin"]):
            states, captures, poses = {}, {}, {}
            for name in STREAMS:
                frame = plan["views"][name][number]
                states[name] = policy.mapper.pose_at(frame["timestamp_ns"], times, samples, 0)
                with np.load(args.sim_replay / name / f"frame-{frame['frame_index']:06d}.npz") as saved:
                    captures[name] = {key: saved[key].copy() for key in ("z16", "robot_mask")}
                    poses[name] = saved["pose"].copy()
                poses[name][:3, 3] -= [0, 0, -0.0245]
            policy.set_optical_poses(states, poses)
            scans = {name: policy.scan(name, cap["z16"], states[name]) for name, cap in captures.items()}
            observed = ObservedEngine()
            policy.fuse(scans, states["cabin"]["slew_yaw_rad"], observed)
            for name, (points, _) in zip(STREAMS, observed.inputs):
                info = policy.projector.metadata[STREAMS[name]]
                intr = info["depth_intrinsics"]
                u = np.rint(intr["fx"] * points[:, 0] / points[:, 2] + intr["ppx"]).astype(int)
                v = np.rint(intr["fy"] * points[:, 1] / points[:, 2] + intr["ppy"]).astype(int)
                if info["rotation_deg"] == 180:
                    u, v = intr["width"] - 1 - u, intr["height"] - 1 - v
                total += len(points)
                leaked += int(captures[name]["robot_mask"][v, u].sum())
            for name, scan in scans.items():
                stride = policy.projection.stride
                cad = captures[name]["robot_mask"][::stride, ::stride]
                valid_ground = (scan[4] > 0) & ~cad
                ground += int(valid_ground.sum())
                ground_rejected += int((valid_ground & (scan[4] == 2)).sum())
        results[profile] = {
            "fused_points": total,
            "cad_robot_points_retained": leaked,
            "cad_leak_pct_of_fused": 100 * leaked / max(1, total),
            "non_cad_returns_masked_pct": 100 * ground_rejected / max(1, ground),
        }
        print("CAD check:", profile, results[profile], flush=True)
    report["simulated_cad_reference"] = results


def run(args):
    args.robot_repo = robot_checkout(args.robot_repo)
    args.fastdem = fastdem_binary(args.fastdem)
    source = json.loads((args.dataset / f"native/episode-{args.episode:06d}/source.json").read_text())
    table = pq.read_table(
        args.dataset / "data/chunk-000/file-000.parquet",
        columns=["frame_index", "clock.depth_host_ns"],
        filters=[("episode_index", "=", args.episode)],
    ).to_pydict()
    selected = [
        i
        for i, index in enumerate(table["frame_index"])
        if index >= args.start and (args.stop is None or index < args.stop) and (index - args.start) % args.stride == 0
    ]
    indices = [table["frame_index"][i] for i in selected]
    readers = {
        name: selected_depth_frames(
            args.dataset / f"native/episode-{args.episode:06d}/{key}.depth.zst", indices, (480, 640)
        )
        for name, key in STREAMS.items()
    }
    times, poses = pose_series(args.dataset, args.episode)
    policies = {
        name: RobotPolicy(
            args.robot_repo, args.dataset, source["session"], empirical_correction=True, overrides=overrides
        )
        for name, overrides in PROFILES.items()
    }
    axis = 0.9 - (np.arange(64) + 0.5) * 0.0125
    lateral = 0.4 - (np.arange(64) + 0.5) * 0.0125
    roi = (axis[:, None] >= 0.2) & (axis[:, None] <= 0.65) & (np.abs(lateral)[None, :] <= 0.25)
    common = {name: Moments((64, 64)) for name in policies}
    own = {name: Moments((64, 64)) for name in policies}
    plane = {name: ErrorStats() for name in policies}
    jump = {name: ErrorStats() for name in policies}
    coverage, costs = {name: [] for name in policies}, {name: [] for name in policies}
    previous, accepted = None, 0
    args.out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        engines = {
            name: ObservedEngine(stack.enter_context(policy.engine(args.fastdem))) for name, policy in policies.items()
        }
        for sample, row in enumerate(selected):
            images = {name: next(reader) for name, reader in readers.items()}
            stamps = table["clock.depth_host_ns"][row]
            if any(stamp <= 0 or stamp < times[0] or stamp > times[-1] for stamp in stamps):
                continue
            states = {
                name: next(iter(policies.values())).mapper.pose_at(stamp, times, poses, 0)
                for name, stamp in zip(STREAMS, stamps)
            }
            observations = {}
            for name, policy in policies.items():
                started = time.perf_counter()
                scans = {camera: policy.scan(camera, images[camera], states[camera]) for camera in STREAMS}
                policy.fuse(scans, states["cabin"]["slew_yaw_rad"], engines[name])
                costs[name].append(1000 * (time.perf_counter() - started))
                world = np.concatenate([p @ t[:, :3].T + t[:, 3] for p, t in engines[name].inputs])
                observations[name] = policy.mapper.raster(
                    world, 0
                )  # fixed undercarriage grid; yaw does not move the ROI
                known = np.isfinite(observations[name]) & roi
                coverage[name].append(100 * known.sum() / roi.sum())
                own[name].add(observations[name], roi)
            shared = np.logical_and.reduce([np.isfinite(value) for value in observations.values()]) & roi
            for name, values in observations.items():
                common[name].add(values, shared)
                plane[name].add(np.where(shared, values * 1000, np.nan), int(shared.sum()), int(shared.sum()))
                if previous is not None:
                    use = shared & previous["shared"]
                    jump[name].add(
                        np.where(use, (values - previous[name]) * 1000, np.nan), int(use.sum()), int(use.sum())
                    )
            previous = {**observations, "shared": shared}
            accepted += 1
            if sample % 50 == 0:
                print(f"Filter sweep: {sample + 1}/{len(selected)} samples", flush=True)
    repeated = next(iter(common.values())).count >= 5
    if not repeated.any():
        raise ValueError("No repeated common ROI cells; choose a longer flat recording")
    report = {
        "episode": args.episode,
        "samples": accepted,
        "profiles": PROFILES,
        "roi": {"frame": "fixed level undercarriage", "x_m": [0.2, 0.65], "y_m": [-0.25, 0.25]},
        "common_repeated_cells": int(repeated.sum()),
        "provenance": policies["current_wide"].provenance(),
        "method": (
            "Identical cell/time support for temporal spread and plane residual; first compute sample excluded; "
            "no rendering, video or disk writes in timings"
        ),
        "frame_range": [indices[0], indices[-1]],
        "stride": args.stride,
        "dataset": str(args.dataset),
        "results": {},
    }
    for name in policies:
        report["results"][name] = {
            "coverage_pct": float(np.mean(coverage[name])),
            "ever_observed_roi_pct": float(100 * (own[name].count > 0)[roi].mean()),
            "common_temporal_sigma_p50_mm": float(np.median(common[name].sigma()[repeated])),
            "common_temporal_sigma_p90_mm": float(np.percentile(common[name].sigma()[repeated], 90)),
            "flat_plane_error": plane[name].result(),
            "consecutive_observation_change": jump[name].result(),
            "compute_p50_ms": float(np.median(costs[name][1:])),
            "compute_p95_ms": float(np.percentile(costs[name][1:], 95)),
        }
    if args.sim_replay:
        assess_sim(args, report)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    draw_report(args.out, report["results"])
    print(json.dumps(report["results"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--robot-repo", type=Path)
    parser.add_argument("--fastdem", type=Path)
    parser.add_argument("--episode", type=int, default=0, help="Use a stationary flat-sand take for a noise comparison")
    parser.add_argument("--start", type=int, default=12)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument(
        "--sim-replay", type=Path, help="Also measure retained CAD returns using a saved simulation comparison"
    )
    parser.add_argument("--out", type=Path, default=Path("logs/filter_sweep"))
    args = parser.parse_args()
    if args.stride < 1 or args.start < 0 or (args.stop is not None and args.stop <= args.start):
        parser.error("Expected a positive stride and nonempty frame range")
    run(args)


if __name__ == "__main__":
    main()
