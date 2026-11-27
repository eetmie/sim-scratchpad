"""Check that a synthetic IR calibration survives fitting and USD composition."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pxr import Gf, Usd

from . import BASE_STAGE
from .fit_mount import errors, fit
from .usd_cameras import apply, inspect, pose


@pytest.mark.parametrize(
    "linear",
    [np.diag([1.001, 1, 1]), [[1, 0.001, 0], [0, 1, 0], [0, 0, 1]], np.diag([-1, 1, 1])],
    ids=["scale", "shear", "reflection"],
)
def test_camera_pose_rejects_nonrigid_transforms(linear):
    transform = np.eye(4)
    transform[:3, :3] = linear
    with pytest.raises(ValueError, match="Camera pose must be rigid"):
        pose(Gf.Matrix4d(*transform.ravel()), 1.0)


def test_native_ir_fit_and_overlay(tmp_path: Path) -> None:
    original_bytes = BASE_STAGE.read_bytes()
    base = inspect(Usd.Stage.Open(str(BASE_STAGE)), None)
    width, height = 640, 480
    k = np.array([[410.0, 0, 328.0], [0, 394.0, 231.0], [0, 0, 1]])
    d = [0.12, -0.025, 0.001, -0.002, 0.003]
    for key, rotation in (("cam1", 180), ("cam2", 0)):
        record = {
            "rotation_deg": rotation,
            "ir_intrinsics": {
                "width": width,
                "height": height,
                "fx": k[0, 0],
                "fy": k[1, 1],
                "ppx": k[0, 2],
                "ppy": k[1, 2],
                "coeffs": d,
                "model": "Brown Conrady",
            },
        }
        (tmp_path / f"{key}.depth.json").write_text(json.dumps(record))

    seeded = inspect(Usd.Stage.Open(str(BASE_STAGE)), tmp_path)["cameras"]["arm"]
    initial_mount = seeded["mount"]
    to_link = Rotation.from_quat(initial_mount["quaternion_xyzw"]) * Rotation.from_euler(
        "xyz", [6, -4, 2], degrees=True
    )
    xyz = np.array(initial_mount["translation_m"]) + [0.012, -0.007, 0.018]
    rng = np.random.default_rng(435)
    optical = rng.uniform([-0.22, -0.16, 0.48], [0.22, 0.16, 0.94], (40, 3))
    points = to_link.apply(optical) + xyz
    raw_pixels, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), k, np.array(d))
    pixels = np.array([width - 1, height - 1]) - raw_pixels[:, 0]
    mount, inliers = fit(seeded, points[:24], pixels[:24])
    assert inliers == 24
    assert errors(seeded, points[24:], pixels[24:])["rms_px"] > 10

    overlay = tmp_path / "calibrated.usda"
    apply(BASE_STAGE, {"arm": {**seeded, "mount": mount}}, overlay)
    actual = inspect(Usd.Stage.Open(str(overlay)), None)
    camera = actual["cameras"]["arm"]
    assert errors(camera, points[24:], pixels[24:])["rms_px"] < 1e-4
    np.testing.assert_allclose(camera["mount"]["translation_m"], xyz, atol=1e-6)
    np.testing.assert_allclose(
        Rotation.from_quat(camera["mount"]["quaternion_xyzw"]).as_matrix(), to_link.as_matrix(), atol=1e-6
    )
    assert actual["meters_per_unit"] == 1
    assert actual["up_axis"] == "Z"
    assert actual["gaussian_splats"] == ["/Environment/_dgs_40k_1m_sh3"]
    assert actual["cameras"]["cabin"] == base["cameras"]["cabin"]
    assert BASE_STAGE.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="new .usda"):
        apply(BASE_STAGE, {"arm": seeded}, overlay)
