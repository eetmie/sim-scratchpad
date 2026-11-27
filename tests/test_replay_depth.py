"""Native depth framing and comparison units, validity and error sign."""

import numpy as np
import pytest
import zstandard
from replay_depth import ErrorStats, depth_error, selected_depth_frames


def test_fastdem_archive_preserves_optical_points_and_sensor_origin():
    import io
    import struct

    from offline_robot_maps import ScanArchive

    stream = io.BytesIO()
    transform = np.array([[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3]])
    assert ScanArchive(stream).update((np.array([[1, 2, 3]]), transform)) == (None, None)
    raw = stream.getvalue()
    assert len(raw) == 116
    assert struct.unpack_from("<II", raw) == (1, 1)
    assert struct.unpack_from("<3f", raw, 104) == (1, 2, 3)
    assert struct.unpack_from("<d", raw, 32)[0] == 0.1
    assert struct.unpack_from("<d", raw, 64)[0] == 0.2
    assert struct.unpack_from("<d", raw, 96)[0] == 0.3


def test_native_zstd_frame_selection_and_truncation(tmp_path):
    frames = [np.full((2, 3), value, dtype="<u2") for value in (11, 27, 93)]
    path = tmp_path / "cam1.depth.zst"
    compressor = zstandard.ZstdCompressor()
    path.write_bytes(b"".join(compressor.compress(frame.tobytes()) for frame in frames))
    selected = list(selected_depth_frames(path, [0, 2], (2, 3)))
    assert len(selected) == 2
    np.testing.assert_array_equal(selected[0], np.full((2, 3), 11))
    np.testing.assert_array_equal(selected[1], np.full((2, 3), 93))
    # A partial final image must not silently shift the depth-to-pose pairing.
    path.write_bytes(compressor.compress(frames[0].tobytes() + b"\x01\x00"))
    with pytest.raises(ValueError, match="incomplete depth frame 1"):
        list(selected_depth_frames(path, [1], (2, 3)))


def test_recorded_orientation_radial_validity_and_signed_optical_error():
    sensor = {
        "rotation_deg": 180,
        "depth_scale_m": 0.0001,
        "depth_intrinsics": {"fx": 1, "fy": 1, "ppx": 0, "ppy": 0},
    }
    # Rotated last pixel is the optical centre. The first pixel is outside radial
    # range even though its optical Z is < 0.9 m; zero has no measurement.
    real = np.array([[8000, 0, 4000]], dtype=np.uint16)
    sim = np.array([[8000, 3000, 4100]], dtype=np.uint16)
    error, rv, sv = depth_error(real, sim, sensor)
    np.testing.assert_array_equal(rv, [[False, False, True]])
    np.testing.assert_array_equal(sv, [[False, True, True]])
    assert np.isnan(error[0, :2]).all()
    assert error[0, 2] == pytest.approx(10, abs=0.001)
    stats = ErrorStats()
    stats.add(np.array([-10.0, 20.0, np.nan]), 3, 4)
    result = stats.result()
    assert result["common_pixels"] == 2
    assert result["real_overlap_pct"] == pytest.approx(200 / 3)
    assert result["mae_mm"] == 15
    assert result["bias_sim_minus_real_mm"] == 5
    assert result["rmse_mm"] == pytest.approx(np.sqrt(250))
