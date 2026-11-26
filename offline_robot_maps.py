"""Apply kaivuriprokkis's mapping masks to paired real/sim depth without rerendering.

Exports current front maps and binary inputs for the robot's ROS-free FastDEM
adapter. Surround maps are generated only when --fastdem is supplied.
"""

import argparse
import hashlib
import json
import logging
import struct
import subprocess
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

import numpy as np

from replay_depth import ErrorStats, depth_error, selected_depth_frames
from robot_mapping import STREAMS, RobotPolicy, millimetres

logger = logging.getLogger(__name__)


class ScanArchive:
    """Archive the real FastDEM wire protocol, optionally forwarding to a running core."""

    def __init__(self, stream, engine=None):
        self.stream, self.engine = stream, engine

    def update(self, *inputs):
        self.stream.write(struct.pack("<I", len(inputs)))
        for points, transform in inputs:
            self.stream.write(struct.pack("<I", len(points)))
            self.stream.write(np.asarray(transform, dtype="<f8").tobytes())
            self.stream.write(np.asarray(points, dtype="<f4").tobytes())
        return self.engine.update(*inputs) if self.engine is not None else (None, None)


class ReviewVideo:
    """Recorded-time review of actual robot-policy masks and resulting maps."""

    def __init__(self, out, frames):
        import av
        from matplotlib import colormaps

        self.av, self.out = av, out
        self.frames = frames
        self.start = frames[0]["timestamp_ns"]
        spacing = np.median(np.diff([frame["timestamp_ns"] for frame in frames])) * 1e-9
        self.video = av.open(str(out / "filtered-comparison.mp4"), mode="w")
        self.stream = self.video.add_stream("libx264", rate=Fraction(1 / spacing).limit_denominator(1001))
        self.stream.width, self.stream.height = 1280, 1000
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "veryfast"}
        self.error_lut = (255 * colormaps["coolwarm"](np.linspace(0, 1, 256))[:, :3]).astype(np.uint8)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for packet in self.stream.encode():
            self.video.mux(packet)
        self.video.close()

    def add(self, number, scans, fronts, surrounds):
        import cv2

        page = np.zeros((1000, 1280, 3), np.uint8)
        frame = self.frames[number]

        def label(value, x, y):
            cv2.putText(page, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)

        label(
            f"Robot filters + FastDEM | dataset frame {frame['frame_index']} | "
            f"{(frame['timestamp_ns'] - self.start) * 1e-9:.2f} s",
            12,
            24,
        )
        for column, (name, kind) in enumerate((name, kind) for name in STREAMS for kind in ("real", "sim")):
            depth, mask = scans[kind][name][3:]
            gray = np.rint(255 * np.clip((0.9 - depth) / 0.8, 0, 1)).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            rgb[depth <= 0] = 0
            rgb[mask == 2] = (255, 55, 40)
            page[70:310, column * 320 : (column + 1) * 320] = cv2.resize(
                rgb, (320, 240), interpolation=cv2.INTER_NEAREST
            )
            label(f"{name} {kind}: red = self mask", column * 320 + 8, 57)

        for title, maps, top in (("Front", fronts, 370), ("FastDEM surround", surrounds, 704)):
            if not maps:
                continue
            for column, kind in enumerate(("real", "sim", "error")):
                x = column * 426 + 80
                if kind == "error":
                    height = maps["sim"] - maps["real"]
                    scaled = np.rint(127.5 * (1 + np.clip(np.nan_to_num(height) / 0.05, -1, 1))).astype(np.uint8)
                    rgb = self.error_lut[scaled].copy()
                else:
                    height = maps[kind]
                    gray = np.rint(128 + 112 * np.clip(np.nan_to_num(height) / 0.12, -1, 1)).astype(np.uint8)
                    rgb = np.repeat(gray[..., None], 3, axis=2)
                rgb[~np.isfinite(height)] = 0
                page[top : top + 256, x : x + 256] = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
                label(f"{title}: {kind if kind != 'error' else 'sim - real'}", column * 426 + 50, top - 16)
        label("Height: +/-120 mm, ground gray | Error: +/-50 mm, blue-red | Black: unknown", 12, 985)
        if number in (0, len(self.frames) // 2, len(self.frames) - 1):
            cv2.imwrite(str(self.out / f"filtered-{frame['frame_index']:06d}.png"), page[:, :, ::-1])
        video_frame = self.av.VideoFrame.from_ndarray(page, format="rgb24")
        for packet in self.stream.encode(video_frame):
            self.video.mux(packet)


def review(out: Path, example: dict, fronts: dict, surrounds: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for row, name in enumerate(STREAMS):
        for col, kind in enumerate(("real", "sim")):
            scan = example[kind][name]
            depth, mask = scan[3:]
            ax = axes[row, col]
            ax.imshow(np.where(depth > 0, depth, np.nan), cmap="viridis", vmin=0.1, vmax=0.9)
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask == 2] = (1, 0.1, 0.1, 0.85)
            ax.imshow(overlay)
            ax.set(title=f"{name} {kind}: robot policy rejects in red", xlabel="Strided stored pixel")
    fig.savefig(out / "masks-middle.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, kind in zip(axes, ("real", "sim")):
        im = ax.imshow(fronts[kind], vmin=-0.12, vmax=0.12, cmap="coolwarm")
        ax.set(title=f"Masked {kind} front height", xlabel="Column decreases Y", ylabel="Row decreases X")
        fig.colorbar(im, ax=ax, label="Height (m)")
    im = axes[2].imshow((fronts["sim"] - fronts["real"]) * 1000, vmin=-50, vmax=50, cmap="coolwarm")
    axes[2].set(title="Sim minus real", xlabel="Column decreases Y")
    fig.colorbar(im, ax=axes[2], label="Height difference (mm)")
    fig.savefig(out / "fronts-middle.png", dpi=140)
    plt.close(fig)
    if surrounds:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, kind in zip(axes, ("real", "sim")):
            im = ax.imshow(surrounds[kind], vmin=-0.12, vmax=0.12, cmap="coolwarm")
            ax.set(title=f"{kind.capitalize()} FastDEM surround", xlabel="Column decreases Y", ylabel="Row decreases X")
            fig.colorbar(im, ax=ax, label="Height (m)")
        im = axes[2].imshow((surrounds["sim"] - surrounds["real"]) * 1000, vmin=-50, vmax=50, cmap="coolwarm")
        axes[2].set(title="Sim minus real", xlabel="Column decreases Y")
        fig.colorbar(im, ax=axes[2], label="Height difference (mm)")
        fig.savefig(out / "surrounds-middle.png", dpi=140)
        plt.close(fig)


def run(args) -> None:
    from camera_calibration.recorded_poses import NATIVE_TO_STAGE

    native = args.robot_repo / "native_stack"
    plan = json.loads((args.replay / "replay.json").read_text())
    source = json.loads((args.dataset / f"native/episode-{plan['episode']:06d}/source.json").read_text())
    policies = {
        kind: RobotPolicy(args.robot_repo, args.dataset, source["session"], empirical_correction=kind == "real")
        for kind in ("real", "sim")
    }
    mapper = policies["sim"].mapper
    policy, projection = policies["sim"].policy, policies["sim"].projection
    real_cal = policies["real"].calibration
    samples = {}
    with (args.dataset / f"native/episode-{plan['episode']:06d}/samples.jsonl").open() as stream:
        for line in stream:
            st = json.loads(line)["status"]
            if st["fk_valid"] and st["imu_healthy"]:
                samples[st["imu_host_monotonic_ns"]] = st
    times = np.array(sorted(samples), np.int64)
    poses = [samples[t] for t in times]
    indices = [f["frame_index"] for f in plan["views"]["cabin"]]
    readers = {
        name: selected_depth_frames(
            args.dataset / f"native/episode-{plan['episode']:06d}/{key}.depth.zst", indices, (480, 640)
        )
        for name, key in STREAMS.items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "-C", str(args.robot_repo), "rev-parse", "HEAD"], text=True).strip()
    inputs = [
        native / "tools/heightmap_bench/mapper.py",
        native / "tools/sand_height.py",
        native / "config/heightmap_pipeline.json",
        native / "config/arm_model.npz",
        native / "tools/heightmap_bench/fastdem.yaml",
    ]
    settings = {
        "robot_revision": revision,
        "pipeline": policy,
        "calibration": real_cal,
        "pose_clock": "imu_host_monotonic_ns, matching camera-calibration replay",
        "simulation": "Saved actual optical poses; empirical depth correction disabled; raw Z16 unchanged",
        "surround_generated": args.fastdem is not None,
        "fastdem_input": "little-endian optical points plus 3x4 sensor poses, including origins for ray clearing",
        "policy_provenance": "Current robot policy; the dataset's full original projection settings are unknown",
        "source_sha256": {
            str(path.relative_to(args.robot_repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
        },
        "fastdem_executable": str(args.fastdem.resolve()) if args.fastdem else None,
        "map_age_units": "accepted paired map updates; 65535 means unknown",
        "map_start": "Empty at the beginning of this selected replay segment",
        "raw_depth_error": "Uncorrected optical Z outside the union of real/sim self masks",
    }
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    config = args.robot_repo / "native_stack/tools/heightmap_bench/fastdem.yaml"
    (args.out / "fastdem.yaml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")

    depth_stats = {name: ErrorStats() for name in STREAMS}
    front_stats = ErrorStats()
    surround_stats = ErrorStats()
    diagnostics = []
    with ExitStack() as stack:
        video = stack.enter_context(ReviewVideo(args.out, plan["views"]["cabin"])) if args.video else None
        engines = {}
        for kind in ("real", "sim"):
            process = stack.enter_context(mapper.FastDEM(args.fastdem, config)) if args.fastdem else None
            engines[kind] = ScanArchive(stack.enter_context((args.out / f"{kind}.scans").open("wb")), process)
        for number, index in enumerate(indices):
            scans = {kind: {} for kind in engines}
            frame_diagnostics = {"frame_index": index}
            states, rendered, optical_poses = {}, {}, {}
            for name, key in STREAMS.items():
                frame = plan["views"][name][number]
                states[name] = mapper.pose_at(frame["timestamp_ns"], times, poses, 0)
                with np.load(args.replay / name / f"frame-{index:06d}.npz") as saved:
                    rendered[name], optical_poses[name] = saved["z16"].copy(), saved["pose"].copy()
                optical_poses[name][:3, 3] -= NATIVE_TO_STAGE
            policies["sim"].set_optical_poses(states, optical_poses)
            yaw = states["cabin"]["slew_yaw_rad"]
            for name, key in STREAMS.items():
                measured, simulated = next(readers[name]), rendered[name]
                for kind, image in (("real", measured), ("sim", simulated)):
                    scan = policies[kind].scan(name, image, states[name])
                    scans[kind][name] = scan
                    mask = scan[4]
                    frame_diagnostics[f"{name}_{kind}_reject_pct"] = float(
                        100 * np.sum(mask == 2) / max(1, np.sum(mask > 0))
                    )
                # Compare the raw optical depth outside the union of both self masks.
                error, rv, sv = depth_error(measured, simulated, policies["real"].projector.metadata[key])
                stride = projection.stride
                exclude = (scans["real"][name][4] == 2) | (scans["sim"][name][4] == 2)
                err = error[::stride, ::stride].copy()
                err[exclude] = np.nan
                depth_stats[name].add(
                    err, int((rv[::stride, ::stride] & ~exclude).sum()), int((sv[::stride, ::stride] & ~exclude).sum())
                )
            fronts, surrounds, arrays = (
                {},
                {},
                {
                    "frame_index": np.array(index),
                    "depth_host_ns": np.array(
                        [plan["views"][name][number]["timestamp_ns"] for name in STREAMS], dtype=np.int64
                    ),
                },
            )
            for kind, engine in engines.items():
                front, surround, age = policies[kind].fuse(scans[kind], yaw, engine)
                fronts[kind] = front
                arrays[f"front_{kind}_mm"] = millimetres(front)
                if surround is not None:
                    surrounds[kind] = surround
                    arrays[f"surround_{kind}_mm"] = millimetres(surround)
                    arrays[f"surround_{kind}_age"] = age
                for name, scan in scans[kind].items():
                    arrays[f"mask_{kind}_{name}"] = scan[4]
            real_front, sim_front = fronts["real"], fronts["sim"]
            front_stats.add(
                (sim_front - real_front) * 1000, int(np.isfinite(real_front).sum()), int(np.isfinite(sim_front).sum())
            )
            if surrounds:
                surround_stats.add(
                    (surrounds["sim"] - surrounds["real"]) * 1000,
                    int(np.isfinite(surrounds["real"]).sum()),
                    int(np.isfinite(surrounds["sim"]).sum()),
                )
            np.savez_compressed(args.out / f"frame-{index:06d}.npz", **arrays)
            diagnostics.append(frame_diagnostics)
            if number == len(indices) // 2:
                review(args.out, scans, fronts, surrounds)
            if video is not None:
                video.add(number, scans, fronts, surrounds)
            if number % 20 == 0 or number == len(indices) - 1:
                logger.info("Robot masking: %d/%d paired samples", number + 1, len(indices))
    summary = {
        "frames": len(indices),
        "robot_revision": revision,
        "front_height_error": front_stats.result(),
        "depth_after_union_self_mask": {name: stats.result() for name, stats in depth_stats.items()},
        "median_reject_pct": {
            key: float(np.median([row[key] for row in diagnostics])) for key in diagnostics[0] if key != "frame_index"
        },
        "surround_generated": args.fastdem is not None,
        "surround_height_error": surround_stats.result() if args.fastdem else None,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.out / "mask_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    logger.info("Saved shared-policy masks, front maps and FastDEM input streams to %s", args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fastdem", type=Path, help="Optional compatible ROS-free FastDEM executable")
    parser.add_argument(
        "--video", action="store_true", help="Write a recorded-time masked depth/map comparison (requires av)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
