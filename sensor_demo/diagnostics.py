"""CSV telemetry, jump-event snapshots, viewport captures and quick-look plots."""

import argparse
import csv
import ctypes
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from soil import bucket_shell, pose_matrix


def capture_viewport(viewer, path):
    """Read the displayed Newton frame without requiring CUDA/OpenGL interop."""
    from PIL import Image

    renderer = viewer.renderer
    gl = renderer.gl
    width, height = renderer._screen_width, renderer._screen_height
    frame = np.empty((height, width, 3), dtype=np.uint8)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, renderer._frame_fbo)
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame.ctypes.data_as(ctypes.c_void_p))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame[::-1]).save(path)


class Recorder:
    """Record solver motion and contact corrections; save state when spikes occur."""

    def __init__(self, directory, soil, capture_every=1.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "config.json").write_text(json.dumps(asdict(soil.cfg), indent=2) + "\n")
        self.stream = (self.directory / "metrics.csv").open("w", newline="")
        self.writer = None
        self.render_stream = (self.directory / "render.csv").open("w", newline="")
        self.render_writer = csv.DictWriter(
            self.render_stream,
            fieldnames=(
                "time_s",
                "frame_ms",
                "surface_ms",
                "surface_triangles",
                "base_max_z_m",
                "display_max_z_m",
                "particle_max_z_m",
            ),
        )
        self.render_writer.writeheader()
        self.capture_every = capture_every
        self.last_capture = self.last_event = -1e9

    def tick(self, soil, pose, sim_time, milliseconds):
        ids = np.flatnonzero(soil.active)
        p, v = soil.positions[ids], soil.velocities[ids]
        speed = np.linalg.norm(v, axis=1)
        horizontal = np.linalg.norm(v[:, :2], axis=1)
        local = (p - pose[:3]) @ pose_matrix(pose)
        away = np.linalg.norm(local, axis=1) > 0.20
        contacts = soil.contact_stats.numpy()[ids] if len(ids) else np.zeros((0, 3))
        raw = soil.raw_speed.numpy()[ids] if len(ids) else np.zeros(0)
        row = dict(
            time_s=sim_time,
            tick_ms=milliseconds,
            particles=len(ids),
            max_speed_m_s=speed.max(initial=0),
            p99_speed_m_s=np.percentile(speed, 99) if len(ids) else 0,
            max_upward_m_s=v[:, 2].max(initial=0),
            max_horizontal_m_s=horizontal.max(initial=0),
            max_observed_speed_m_s=soil.motion_speed[ids].max(initial=0),
            fast_upward_away=int(np.count_nonzero(away & (v[:, 2] > 0.75))),
            fast_horizontal_away=int(np.count_nonzero(away & (horizontal > 0.75))),
            max_raw_speed_m_s=raw.max(initial=0),
            ground_projection_mm=contacts[:, 0].max(initial=0) * 1000,
            bucket_projection_mm=contacts[:, 1].max(initial=0) * 1000,
            grain_projection_mm=contacts[:, 2].max(initial=0) * 1000,
            dynamic_liters=soil.volumes[ids].sum() * 1000,
            volume_error_ml=(soil.volume() - soil.initial_volume) * 1e6,
            max_z_m=p[:, 2].max(initial=soil.cfg.floor),
            pool_full=int(soil.saturated),
        )
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()
        suspicious = (
            row["max_raw_speed_m_s"] > 3
            or row["fast_upward_away"] > 0
            or row["fast_horizontal_away"] > 0
            or row["max_observed_speed_m_s"] > 2.5
        )
        if suspicious and sim_time - self.last_event >= 0.5:
            np.savez_compressed(
                self.directory / f"event_{sim_time:07.3f}.npz",
                positions=p,
                velocities=v,
                observed_speed=soil.motion_speed[ids],
                particle_ids=ids,
                radii=soil.radii[ids],
                volumes=soil.volumes[ids],
                contact_corrections=contacts,
                raw_speed=raw,
                bucket_pose=pose,
                height=soil.height,
                origin=soil.origin,
                cell=soil.cfg.cell,
                time=sim_time,
            )
            self.last_event = sim_time
        return row

    def frame(self, viewer, sim_time, frame_ms=0, surface_ms=0, surface_triangles=0, soil=None, surface=None):
        base = soil.height.max() if soil is not None else float("nan")
        display = surface.display_height.max() if surface is not None else base
        particle = soil.positions[soil.active != 0, 2].max(initial=soil.cfg.floor) if soil else float("nan")
        self.render_writer.writerow(
            dict(
                time_s=sim_time,
                frame_ms=frame_ms,
                surface_ms=surface_ms,
                surface_triangles=surface_triangles,
                base_max_z_m=base,
                display_max_z_m=display,
                particle_max_z_m=particle,
            )
        )
        self.render_stream.flush()
        if viewer is not None and self.capture_every > 0 and sim_time - self.last_capture >= self.capture_every:
            capture_viewport(viewer, self.directory / f"frame_{sim_time:07.3f}.png")
            if surface is not None and soil is not None:
                ids = np.flatnonzero(soil.active)
                np.savez_compressed(
                    self.directory / f"render_{sim_time:07.3f}.npz",
                    positions=soil.positions[ids],
                    velocities=soil.velocities[ids],
                    volumes=soil.volumes[ids],
                    radii=soil.radii[ids],
                    ground_fraction=surface.grounded_fraction[ids],
                    spray=surface.spray_shown[ids],
                    height=soil.height,
                    display_height=surface.display_height,
                    bucket_pose=soil.previous_pose,
                )
            self.last_capture = sim_time

    def close(self):
        self.stream.close()
        self.render_stream.close()


def plot_recording(directory):
    """Generate a standalone numerical summary and a top/side view of the worst event."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    data = np.atleast_1d(np.genfromtxt(directory / "metrics.csv", delimiter=",", names=True))
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t = data["time_s"]
    axes[0].plot(t, data["tick_ms"], label="Soil tick")
    axes[0].axhline(16.67, color="gray", linestyle="--", label="60 Hz budget")
    axes[0].set_ylabel("ms")
    for name in ("max_speed_m_s", "p99_speed_m_s", "max_raw_speed_m_s", "max_horizontal_m_s", "max_observed_speed_m_s"):
        if name not in data.dtype.names:
            continue
        axes[1].plot(t, data[name], label=name)
    axes[1].set_ylabel("m/s")
    for name in ("ground_projection_mm", "bucket_projection_mm", "grain_projection_mm"):
        axes[2].plot(t, data[name], label=name)
    axes[2].set_ylabel("Contact correction (mm)")
    axes[3].plot(t, data["volume_error_ml"], label="Volume drift")
    axes[3].set_ylabel("mL")
    axes[3].set_xlabel("Simulation time (s)")
    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "summary.png", dpi=150)
    plt.close(fig)
    events = list(directory.glob("event_*.npz"))
    if events:

        def event_speed(path):
            with np.load(path) as event:
                return event["raw_speed"].max(initial=0)

        worst = max(events, key=event_speed)
        with np.load(worst) as event:
            p, v = event["positions"], event["velocities"]
            shell, triangles = bucket_shell()
            pose = event["bucket_pose"]
            shell = shell @ pose_matrix(pose).T + pose[:3]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, (a, b), labels in zip(axes, ((0, 1), (0, 2)), (("x", "y"), ("x", "z"))):
                for triangle in triangles.reshape(-1, 3):
                    polygon = shell[np.append(triangle, triangle[0])]
                    ax.plot(polygon[:, a], polygon[:, b], color="black", linewidth=0.4, alpha=0.4)
                points = ax.scatter(p[:, a], p[:, b], c=v[:, 2], s=5, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set(xlabel=labels[0] + " (m)", ylabel=labels[1] + " (m)", aspect="equal")
            height = event["height"]
            row = np.clip(int((pose[1] - event["origin"]) / event["cell"]), 0, height.shape[0] - 1)
            xs = event["origin"] + (np.arange(height.shape[1]) + 0.5) * event["cell"]
            axes[1].plot(xs, height[row], color="saddlebrown", label="Resting surface at bucket center")
            axes[1].legend()
            fig.colorbar(points, ax=axes, label="Vertical speed (m/s)")
            fig.suptitle(f"Largest velocity spike: {worst.name}")
            fig.savefig(directory / "worst_event.png", dpi=150)
            plt.close(fig)
    print(f"Summary written to {directory / 'summary.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Demo --record_dir to summarize.")
    plot_recording(parser.parse_args().directory)
