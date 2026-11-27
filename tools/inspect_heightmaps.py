"""Summarize an exported depth/map capture; optionally save a preview without starting Isaac Sim."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Directory containing profile.json and latest.npz")
    parser.add_argument("--plot", action="store_true", help="Write preview.png in the capture directory")
    args = parser.parse_args()
    profile = json.loads((args.capture / "profile.json").read_text(encoding="utf-8"))
    panels = []
    with np.load(args.capture / "latest.npz", allow_pickle=False) as data:
        print("Arrays:", {key: (data[key].shape, str(data[key].dtype)) for key in data.files})
        for name in ("cabin", "arm"):
            z16 = data[f"depth_{name}_z16"]
            depth = z16 * profile["sensors"][name]["depth_scale_m"]
            panels.append((f"{name.title()} optical depth (m)", depth, z16 != 0, 0.1, 0.9, "viridis"))
        for name in ("front", "surround"):
            mm = data[f"{name}_mm"]
            panels.append((f"{name.title()} height (mm)", mm, mm != -32768, -120, 120, "coolwarm"))
    for title, values, valid, *_ in panels:
        quantiles = np.percentile(values[valid], [10, 50, 90]).tolist() if valid.any() else []
        print(f"{title}: observed={int(valid.sum())}/{valid.size}, p10/p50/p90={quantiles}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
        for ax, (title, values, valid, low, high, cmap) in zip(axes.flat, panels, strict=True):
            im = ax.imshow(np.where(valid, values, np.nan), vmin=low, vmax=high, cmap=cmap)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        destination = args.capture / "preview.png"
        fig.savefig(destination, dpi=140)
        plt.close(fig)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
