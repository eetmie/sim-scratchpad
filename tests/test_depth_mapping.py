"""Recorded D435i and terrain-image frame/unit contracts, independent of Kit."""

from unittest.mock import Mock

import numpy as np
import pytest
from depth_mapping import (
    UNKNOWN_AGE,
    UNKNOWN_MM,
    DepthMapping,
    TerrainMapper,
    deproject,
    level_points,
    map_image,
    optical_pose,
    recorded_depth,
)


@pytest.mark.parametrize("cuda", [False, True])
def test_panel_upload_preserves_pixels_and_size_after_resize(cuda):
    panel = DepthMapping.__new__(DepthMapping)
    panel.panel_device = None
    if cuda:
        wp = pytest.importorskip("warp")
        if not wp.is_cuda_available():
            pytest.skip("CUDA device required for GPU panel upload")
        panel.panel_device = wp.get_device("cuda:0")
    provider = Mock()
    panel.providers, panel.image_buffers = {"arm": provider}, {}
    gray = np.arange(15, dtype=np.uint8).reshape(3, 5)
    for rgb in (False, True):
        image = np.stack((gray, gray + 10, gray + 20), axis=-1)[::2, ::2] if rgb else gray
        panel._image("arm", image)
        buffer = panel.image_buffers["arm"]
        size = [image.shape[1], image.shape[0]]
        if cuda:
            provider.set_bytes_data_from_gpu.assert_called_with(buffer.ptr, size)
            rgba = buffer.numpy()
        else:
            data, actual_size = provider.set_data_array.call_args.args
            assert actual_size == size
            rgba = data.reshape(*image.shape[:2], 4)
        np.testing.assert_array_equal(rgba[..., :3], image if rgb else np.repeat(gray[..., None], 3, axis=-1))
        assert np.all(rgba[..., 3] == 255)


def test_depth_quantization_owns_rotated_pixels_and_preserves_invalid_and_saturation_rules():
    samples = np.array([[np.nan, np.inf, -1, 0], [2.5, 3.5, 65535.75, 65536]], dtype=np.float32)
    samples.setflags(write=False)
    z16 = recorded_depth(samples, {"depth_scale_m": 1.0, "rotation_deg": 180})
    assert z16.dtype == np.uint16 and z16.flags.c_contiguous
    np.testing.assert_array_equal(z16, [[0, 65535, 4, 2], [0, 0, 0, 0]])


def test_depth_to_ir_baseline_is_rotated_into_parent_frame():
    q = np.sqrt(0.5)
    mount = {"translation_m": [1, 2, 3], "quaternion_xyzw": [0, 0, q, q]}
    sensor = {
        "depth_to_ir_extrinsics": {
            "rotation_column_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "translation_m": [-0.05, 0, 0],
        }
    }
    pose = optical_pose(mount, sensor)
    np.testing.assert_allclose(pose[:3, 3], [1, 1.95, 3], atol=1e-12)
    np.testing.assert_allclose(pose[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


def test_rotated_depth_uses_optical_z_off_center_intrinsics_and_radial_range():
    sensor = {"rotation_deg": 180, "depth_intrinsics": {"width": 5, "height": 3, "fx": 4, "fy": 4, "ppx": 0, "ppy": 0}}
    # The final 0.8 m pixel is inside the optical-Z limit but outside the radial limit.
    unrotated = np.array([[0.5, 0, 0, 0, 2], [0, 0, 0, 0, 0], [np.nan, 0, 0.4, 0, 0.8]])
    pose = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    points = deproject(unrotated[::-1, ::-1], sensor, pose, stride=2)
    np.testing.assert_allclose(points, [[1, 2, 3.5], [0.8, 2.2, 3.4]])
    local = level_points(points, np.array([1, 2, 100]), np.pi / 2, 3.4)
    np.testing.assert_allclose(local, [[0, 0, 0.1], [0.2, 0.2, 0]], atol=1e-12)


def test_dataset_map_layout_median_unknown_history_and_age():
    profile = {
        "front": {"x_m": [0.1, 0.9], "y_m": [-0.4, 0.4], "cell_m": 0.0125},
        "surround": {"xy_m": [-0.8, 0.8], "cell_m": 0.01, "radius_m": [0.14, 0.8]},
        "fps": 30,
    }
    mapper = TerrainMapper(profile)
    front = np.array([[0.30625, 0.10625, 0.005], [0.30625, 0.10625, 0.015]])
    base = np.array([[0.205, 0.105, 0.02], [0.01, 0.01, 0.1], [1, 0, 0.1]])
    first = mapper.update(front, base, 0)
    assert first["front_mm"].shape == (64, 64)
    assert first["front_mm"][47, 23] == 10
    assert np.count_nonzero(first["front_mm"] != UNKNOWN_MM) == 1
    assert first["surround_mm"].shape == (160, 160)
    assert first["surround_mm"][59, 69] == 20
    assert np.count_nonzero(first["surround_mm"] != UNKNOWN_MM) == 1
    second = mapper.update(np.empty((0, 3)), np.empty((0, 3)), 0.2)
    assert np.all(second["front_mm"] == UNKNOWN_MM)
    assert second["surround_mm"][59, 69] == 20
    assert second["surround_age_frames"][59, 69] == 6
    assert second["surround_age_frames"][0, 0] == UNKNOWN_AGE
    np.testing.assert_array_equal(map_image(np.array([[UNKNOWN_MM, -120, 0, 120]])), [[0, 16, 128, 240]])
