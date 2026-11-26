"""Fit an IR camera mount from known parent-link points and corresponding recorded pixels."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .usd_cameras import intrinsics

CSV_HEADER = "x_m,y_m,z_m,u_px,v_px"


def correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as file:
        if file.readline().strip() != CSV_HEADER:
            raise ValueError(f"{path}: expected {CSV_HEADER}")
        rows = np.loadtxt(file, delimiter=",", ndmin=2)
    if rows.shape[1] != 5 or not np.isfinite(rows).all():
        raise ValueError(f"{path}: expected finite XYZ and UV coordinates")
    return rows[:, :3].copy(), rows[:, 3:].copy()


def recorded_pixels(camera: dict, pixels: np.ndarray) -> np.ndarray:
    """180° is its own inverse, so this converts in either direction."""
    rotation = camera["image_rotation_deg"]
    if rotation == 180:
        lens = camera["intrinsics"]
        return np.array([lens["width"] - 1, lens["height"] - 1]) - pixels
    if rotation != 0:
        raise ValueError("image_rotation_deg must be 0 or 180")
    return pixels


def project(camera: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k, d = intrinsics(camera["intrinsics"])
    mount = camera["mount"]
    to_optical = Rotation.from_quat(mount["quaternion_xyzw"]).inv()
    optical = to_optical.apply(points - mount["translation_m"])
    pixels, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), k, d)
    return recorded_pixels(camera, pixels[:, 0]), optical[:, 2] > 0


def fit(camera: dict, points: np.ndarray, pixels: np.ndarray) -> tuple[dict, int]:
    if len(points) < 6 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("At least six points spread across the target are required")
    k, d = intrinsics(camera["intrinsics"])
    sensor_pixels = recorded_pixels(camera, pixels)
    ok, rv, tv, inliers = cv2.solvePnPRansac(points, sensor_pixels, k, d, reprojectionError=3.0)
    if not ok or len(inliers) < 6:
        raise ValueError("Could not fit the mount with at least six inliers within three pixels")
    selected = inliers.ravel()
    rv, tv = cv2.solvePnPRefineLM(points[selected], sensor_pixels[selected], k, d, rv, tv)
    to_link = Rotation.from_rotvec(rv.ravel()).inv()
    if np.any((to_link.inv().apply(points[selected]) + tv.ravel())[:, 2] <= 0):
        raise ValueError("Fitted mount places target points behind the camera")
    return {
        "translation_m": to_link.apply(-tv.ravel()).tolist(),
        "quaternion_xyzw": to_link.as_quat().tolist(),
    }, len(selected)


def errors(camera: dict, points: np.ndarray, observed: np.ndarray) -> dict:
    predicted, visible = project(camera, points)
    error = np.linalg.norm(predicted[visible] - observed[visible], axis=1)
    return {
        "points": len(points),
        "in_front": int(visible.sum()),
        "rms_px": float(np.sqrt(np.mean(error**2))) if error.size else None,
        "p90_px": float(np.percentile(error, 90)) if error.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("camera", choices=("cabin", "arm"))
    parser.add_argument("fit_csv", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.calibration.read_text())
    baseline = data["cameras"][args.camera]
    train = correspondences(args.fit_csv)
    mount, inliers = fit(baseline, *train)
    fitted = {**baseline, "mount": mount}
    fitted.pop("optical_to_world", None)
    data["cameras"][args.camera] = fitted
    summary = {"inliers": inliers, "fit": {"before": errors(baseline, *train), "after": errors(fitted, *train)}}
    if args.holdout:
        holdout = correspondences(args.holdout)
        summary["holdout"] = {"before": errors(baseline, *holdout), "after": errors(fitted, *holdout)}
    with args.out.open("x") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
