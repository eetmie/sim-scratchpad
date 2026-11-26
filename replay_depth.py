"""Replay logged robot poses at depth exposures and compare D435i depth pixel for pixel."""

import argparse
import csv
import json
import logging
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from depth_mapping import STREAMS, DepthMapping

logger = logging.getLogger(__name__)


def selected_depth_frames(path: Path, indices: list[int], shape: tuple[int, int]):
    """Read selected recorded-orientation Z16 frames from concatenated native zstd frames."""
    import zstandard

    if not indices or indices != sorted(set(indices)) or indices[0] < 0:
        raise ValueError("Depth indices must be nonempty, increasing and nonnegative")
    size = int(np.prod(shape)) * 2
    requested = iter(indices)
    target = next(requested)
    with (
        path.open("rb") as source,
        zstandard.ZstdDecompressor().stream_reader(source, read_across_frames=True) as reader,
    ):
        for index in range(indices[-1] + 1):
            raw = reader.read(size)
            if len(raw) != size:
                raise ValueError(f"{path}: incomplete depth frame {index}; expected {size}, got {len(raw)} bytes")
            if index == target:
                yield np.frombuffer(raw, dtype="<u2").reshape(shape).copy()
                target = next(requested, None)
                if target is None:
                    break


def depth_error(real_z16: np.ndarray, sim_z16: np.ndarray, sensor: dict) -> tuple:
    """Optical-Z residual [mm], plus real/sim validity in the 0.1--0.9 m radial volume.

    Images use the recorder's pixel orientation. No empirical real-depth correction,
    image registration, scale fitting or offset fitting is applied.
    """
    if real_z16.shape != sim_z16.shape:
        raise ValueError("Recorded and rendered depth shapes differ")
    intr = sensor["depth_intrinsics"]
    v, u = np.indices(real_z16.shape)
    ray_length = np.sqrt(1 + ((u - intr["ppx"]) / intr["fx"]) ** 2 + ((v - intr["ppy"]) / intr["fy"]) ** 2)
    if sensor["rotation_deg"] == 180:
        ray_length = ray_length[::-1, ::-1]
    real = real_z16.astype(np.float32) * sensor["depth_scale_m"]
    sim = sim_z16.astype(np.float32) * sensor["depth_scale_m"]
    valid_real = (real > 0) & (real * ray_length >= 0.1) & (real * ray_length <= 0.9)
    valid_sim = (sim > 0) & (sim * ray_length >= 0.1) & (sim * ray_length <= 0.9)
    error_mm = np.where(valid_real & valid_sim, (sim - real) * 1000, np.nan)
    return error_mm, valid_real, valid_sim


class ErrorStats:
    """Pool errors without retaining all images; absolute quantiles have 1 mm bins."""

    def __init__(self):
        self.count = 0
        self.real_count = 0
        self.sim_count = 0
        self.signed_sum = 0.0
        self.abs_sum = 0.0
        self.square_sum = 0.0
        self.abs_hist = np.zeros(2000, dtype=np.int64)

    def add(self, error_mm: np.ndarray, real_count: int, sim_count: int) -> None:
        values = error_mm[np.isfinite(error_mm)].astype(np.float64)
        self.count += values.size
        self.real_count += real_count
        self.sim_count += sim_count
        self.signed_sum += values.sum()
        self.abs_sum += np.abs(values).sum()
        self.square_sum += np.square(values).sum()
        self.abs_hist += np.bincount(np.minimum(np.abs(values).astype(int), 1999), minlength=2000)

    def result(self) -> dict:
        result = {
            "common_pixels": self.count,
            "real_valid_pixels": self.real_count,
            "sim_valid_pixels": self.sim_count,
            "real_overlap_pct": 100 * self.count / self.real_count if self.real_count else None,
            "mae_mm": self.abs_sum / self.count if self.count else None,
            "bias_sim_minus_real_mm": self.signed_sum / self.count if self.count else None,
            "rmse_mm": float(np.sqrt(self.square_sum / self.count)) if self.count else None,
        }
        for name, q in (("median_abs_mm", 0.5), ("p90_abs_mm", 0.9)):
            result[name] = (
                float(np.searchsorted(np.cumsum(self.abs_hist), self.count * q) + 0.5) if self.count else None
            )
        return result


def comparison_image(real: dict, sim: dict, sensors: dict, frame_index: int, elapsed_s: float) -> np.ndarray:
    """Build a labelled depth/error video frame; black pixels have no usable return."""
    import cv2
    from matplotlib import colormaps

    canvas = np.zeros((1080, 1920, 3), np.uint8)
    error_lut = (colormaps["coolwarm"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[:, ::-1]
    for row, name in enumerate(STREAMS):
        error, real_valid, sim_valid = depth_error(real[name], sim[name], sensors[name])
        for col, (image, valid) in enumerate(((real[name], real_valid), (sim[name], sim_valid))):
            metres = image.astype(np.float32) * sensors[name]["depth_scale_m"]
            gray = np.rint(255 * np.clip((metres - 0.1) / 0.8, 0, 1)).astype(np.uint8)
            color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
            color[~valid] = 0
            canvas[90 + row * 510 : 570 + row * 510, col * 640 : (col + 1) * 640] = color
        scaled = np.rint(127.5 * (1 + np.clip(np.nan_to_num(error) / 100, -1, 1))).astype(np.uint8)
        color = error_lut[scaled]
        color[~np.isfinite(error)] = 0
        canvas[90 + row * 510 : 570 + row * 510, 1280:1920] = color
        for col, label in enumerate(("real optical depth", "sim optical depth", "sim - real error")):
            cv2.putText(
                canvas,
                f"{name}: {label}",
                (col * 640 + 12, 82 + row * 510),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        f"Recorded-pose replay | frame {frame_index} | clip time {elapsed_s:.2f} s",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Depth: 0.1-0.9 m (blue-red) | Error: -100 to +100 mm (blue-red) | Black: invalid/out of range",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_report(out: Path, summary: dict, rows: list[dict], plan: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (out / "frames.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained", sharex=True)
    for name in STREAMS:
        selected = [row for row in rows if row["camera"] == name]
        axes[0].plot(
            [r["clip_time_s"] for r in selected], [r["median_abs_mm"] for r in selected], label=f"{name} median"
        )
        axes[0].plot(
            [r["clip_time_s"] for r in selected],
            [r["p90_abs_mm"] for r in selected],
            linestyle="--",
            alpha=0.6,
            label=f"{name} p90",
        )
    axes[0].set(
        ylabel="Absolute optical-depth error (mm)",
        title="Real vs sim: common valid pixels outside the simulated robot mask",
    )
    frames = plan["views"]["cabin"]
    times = [(f["timestamp_ns"] - frames[0]["timestamp_ns"]) * 1e-9 for f in frames]
    for axis, values in zip(
        ("slew", "boom", "arm", "bucket"), np.rad2deg([f["angles_rad"] for f in frames]).T, strict=True
    ):
        axes[1].plot(times, values, label=axis)
    axes[1].set(xlabel="Time from first cabin depth exposure (s)", ylabel="Logged joint angle (degrees)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best", ncol=4)
    fig.savefig(out / "errors_and_angles.png", dpi=150)
    plt.close(fig)


def compare(args, app) -> None:
    import av
    import cv2

    import omni.usd

    from camera_calibration import BASE_STAGE
    from camera_calibration.recorded_poses import apply_transforms, replay_plan

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.dataset / f"native/episode-{args.episode:06d}/source.json").read_text())
    plan = replay_plan(args.dataset, args.episode, args.start, args.stop, args.stride, timestamp_source="depth")
    if any(f["fk_clamped"] for frames in plan["views"].values() for f in frames):
        raise ValueError("Requested replay includes exposures outside the valid FK timestamp range")
    (out / "replay.json").write_text(json.dumps(plan, separators=(",", ":")) + "\n")
    context = omni.usd.get_context()
    if not context.open_stage(str(BASE_STAGE)):
        raise RuntimeError(f"Could not open {BASE_STAGE}")
    stage = context.get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    app.reset_render_settings()
    app.update()
    rig = DepthMapping(
        stage,
        args.dataset,
        out / "sensors",
        panel=args.loop,
        session=source["session"],
        robot_repo=args.robot_repo,
        fastdem=args.fastdem,
        mapping=args.mapping,
        cad_reference=True,
    )
    ops = {}
    readers = {}
    indices = [f["frame_index"] for f in plan["views"]["cabin"]]
    for name, key in STREAMS.items():
        intr = rig.profile["sensors"][name]["depth_intrinsics"]
        readers[name] = selected_depth_frames(
            args.dataset / f"native/episode-{args.episode:06d}/{key}.depth.zst",
            indices,
            (intr["height"], intr["width"]),
        )
        (out / name).mkdir(exist_ok=True)
    totals = {name: {region: ErrorStats() for region in ("all", "outside_sim_robot")} for name in STREAMS}
    rows = []
    first_stamp = plan["views"]["cabin"][0]["timestamp_ns"]
    video = av.open(str(out / "comparison.mp4"), mode="w")
    stream = video.add_stream("libx264", rate=Fraction(rig.profile["fps"], args.stride))
    stream.width, stream.height = 1920, 1080
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "fast"}
    try:
        for number, frame_index in enumerate(indices):
            real, simulated = {}, {}
            for name in STREAMS:
                frame = plan["views"][name][number]
                apply_transforms(stage, frame["transforms"], ops)
                app.update()
                capture = rig.capture()[name]
                sim = capture["z16"]
                measured = next(readers[name])
                sensor = rig.profile["sensors"][name]
                error, rv, sv = depth_error(measured, sim, sensor)
                outside = ~capture["robot_mask"]
                scene_error = np.where(outside, error, np.nan)
                totals[name]["all"].add(error, int(rv.sum()), int(sv.sum()))
                totals[name]["outside_sim_robot"].add(scene_error, int((rv & outside).sum()), int((sv & outside).sum()))
                stats = ErrorStats()
                stats.add(scene_error, int((rv & outside).sum()), int((sv & outside).sum()))
                rows.append(
                    {
                        "camera": name,
                        "frame_index": frame_index,
                        "timestamp_ns": frame["timestamp_ns"],
                        "clip_time_s": (frame["timestamp_ns"] - first_stamp) * 1e-9,
                        **stats.result(),
                    }
                )
                np.savez_compressed(
                    out / name / f"frame-{frame_index:06d}.npz", **capture, timestamp_ns=np.array(frame["timestamp_ns"])
                )
                real[name], simulated[name] = measured, sim
            elapsed = (plan["views"]["cabin"][number]["timestamp_ns"] - first_stamp) * 1e-9
            image = comparison_image(real, simulated, rig.profile["sensors"], frame_index, elapsed)
            video_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            video_frame.pts = number
            video_frame.time_base = Fraction(args.stride, rig.profile["fps"])
            for packet in stream.encode(video_frame):
                video.mux(packet)
            if number in (0, len(indices) // 2, len(indices) - 1):
                cv2.imwrite(str(out / f"comparison-{frame_index:06d}.png"), image)
            if number % 10 == 0 or number == len(indices) - 1:
                logger.info(
                    "Depth replay: %d/%d paired frames (dataset frame %d)", number + 1, len(indices), frame_index
                )
        for packet in stream.encode():
            video.mux(packet)
        video.close()
        summary = {
            "episode": args.episode,
            "session": source["session"],
            "start": args.start,
            "stop": args.stop,
            "stride": args.stride,
            "paired_frames": len(indices),
            "duration_s": (args.stop - args.start) / rig.profile["fps"],
            "method": "Calibration FK at each depth exposure; native Z16; no depth correction or registration fit",
            "residual": "sim minus real optical Z in millimetres; 0.1-0.9 m radial validity; quantiles use 1 mm bins",
            "joint_names": ["slew", "boom", "arm", "bucket"],
            "logged_angle_range_deg": np.ptp(
                np.rad2deg([f["angles_rad"] for f in plan["views"]["cabin"]]), axis=0
            ).tolist(),
            "cameras": {
                name: {region: stats.result() for region, stats in regions.items()} for name, regions in totals.items()
            },
        }
        write_report(out, summary, rows, plan)
        logger.info("Saved depth comparison, metrics and replay plan to %s", out)
        if args.loop:
            logger.info("Looping logged motion in the viewer. Close Isaac Sim to exit.")
            full = replay_plan(args.dataset, args.episode, args.start, args.stop, 1, timestamp_source="depth")["views"][
                "cabin"
            ]
            started = time.monotonic()
            while app.is_running():
                frame = full[int((time.monotonic() - started) * rig.profile["fps"]) % len(full)]
                apply_transforms(stage, frame["transforms"], ops)
                if not rig.update():
                    app.update()
    finally:
        video.close()
        for reader in readers.values():
            reader.close()
        rig.close()


def main() -> None:
    from robot_mapping import mapping_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--start", type=int, default=12)
    parser.add_argument("--stop", type=int, required=True, help="Exclusive dataset frame index")
    parser.add_argument("--stride", type=int, default=6, help="Compare every Nth recorded frame")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    mapping_arguments(parser)
    parser.add_argument(
        "--loop", action="store_true", help="After comparison, loop the recorded motion in the live viewer"
    )
    args = parser.parse_args()
    if args.episode < 0 or not 0 <= args.start < args.stop or args.stride < 1 or (args.headless and args.loop):
        parser.error("Choose a valid episode/range/stride; --loop requires a visible viewer")
    try:
        import av
    except ImportError:
        parser.error("Video export requires PyAV: uv run --no-sync --with av python replay_depth.py ...")
    av.CodecContext.create("libx264", "w")
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    from launch_3dgs import create_app

    app = create_app(args.headless)
    code = 0
    try:
        compare(args, app)
    except BaseException:
        logger.exception("Recorded-depth replay failed")
        code = 1
        raise
    finally:
        app.close(exit_code=code)


if __name__ == "__main__":
    main()
