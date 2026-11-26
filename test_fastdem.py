"""Exercise the real C++ adapter's binary pipes, map geometry and update ages."""

import io
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from offline_robot_maps import ScanArchive

ROOT = Path(__file__).resolve().parent
EXE = Path(os.environ.get("FASTDEM_EXE", ROOT / "logs/fastdem/build-mingw/fastdem_replay.exe"))


@pytest.mark.skipif(not EXE.is_file(), reason="Build the pinned FastDEM adapter first")
def test_flat_ground_binary_round_trip_and_persistent_age():
    # Optical +Z points down; the sensor is half a metre above a flat surface.
    x, y = np.meshgrid(np.arange(0.205, 0.606, 0.01), np.arange(-0.195, 0.196, 0.01))
    optical = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.5)))
    pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.5]])
    stream = io.BytesIO()
    archive = ScanArchive(stream)
    archive.update((optical, pose))
    archive.update()  # No returns: keep known terrain and advance the accepted-update age.
    result = subprocess.run(
        [str(EXE), str(ROOT / "ros2/fastdem.yaml")],
        input=stream.getvalue(),
        capture_output=True,
        timeout=30,
        check=True,
    )
    size = 160 * 160
    response = size * 6
    assert len(result.stdout) == 2 * response, result.stderr.decode(errors="replace")
    first = np.frombuffer(result.stdout, "<f4", size)
    second = np.frombuffer(result.stdout, "<f4", size, response)
    age_first = np.frombuffer(result.stdout, "<u2", size, size * 4)
    age_second = np.frombuffer(result.stdout, "<u2", size, response + size * 4)
    observed = np.isfinite(first)
    assert observed.sum() > 1000
    np.testing.assert_allclose(first[observed], 0, atol=1e-6)
    np.testing.assert_allclose(second, first, equal_nan=True)
    assert np.all(age_first[observed] == 0)
    assert np.all(age_second[observed] == 1)
    assert np.all(age_second[~observed] == 65535)
