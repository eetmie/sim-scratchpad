"""Save map/depth comparisons and timing plots from a recorded sensor demo."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def inspect(directory):
    directory = Path(directory)
    scans = sorted((directory / "sensors").glob("scan_*.npz"))
    if not scans:
        raise ValueError(f"No sensor snapshots in {directory}")
    times = np.array([float(p.stem[5:]) for p in scans])
    selected = [scans[np.argmin(abs(times - t))] for t in [2, 5, 8, 12, 30, times[-1]]]
    with np.load(selected[0]) as data:
        columns = [("observed", "Virtual lidar height"), ("oracle", "Oracle height")]
        if "depth_cabin_m" in data:
            columns += [("depth_cabin_m", "Cabin depth"), ("depth_arm_m", "Arm depth")]
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(4 * len(columns), 2.6 * len(selected)))
    for row, path in enumerate(selected):
        with np.load(path) as scan:
            for col, (key, label) in enumerate(columns):
                ax = axes[row, col]
                depth = key.startswith("depth")
                ax.set_facecolor("#888888")
                im = ax.imshow(
                    scan[key],
                    vmin=0 if depth else -0.3,
                    vmax=2 if depth else 0,
                    cmap="viridis" if depth else "terrain",
                    interpolation="nearest",
                )
                ax.set_title(f"{label}, {float(scan['time_s']):.1f} s")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, label="m", fraction=0.046, pad=0.04)
    fig.suptitle("Gray = unknown. Heights are cabin-local Z. Depth is optical Z.", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(directory / "sensor_comparison.png", dpi=120)
    plt.close(fig)
    data = np.genfromtxt(directory / "sensors/metrics.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(data["time_s"][1:], data["scan_ms"][1:])
    axes[0].set_ylabel("Sensor update ms")
    axes[1].plot(data["time_s"], data["observed_cells"] / 4096 * 100, label="Observed")
    axes[1].plot(data["time_s"], data["oracle_cells"] / 4096 * 100, label="Oracle")
    axes[1].set_ylabel("Coverage %")
    axes[1].legend()
    axes[2].plot(data["time_s"], data["moving_hits"], label="Moving sand")
    axes[2].plot(data["time_s"], data["robot_hits"], label="Occluders")
    axes[2].set_ylabel("Rays")
    axes[2].legend()
    axes[2].set_xlabel("Simulation seconds")
    fig.tight_layout()
    fig.savefig(directory / "sensor_metrics.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    inspect(parser.parse_args().directory)
