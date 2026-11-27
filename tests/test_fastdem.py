"""Exercise the real C++ adapter's binary pipes, map geometry and update ages."""

import io
import os
import subprocess

import numpy as np
import pytest
from offline_robot_maps import ScanArchive
from robot_mapping import ROOT, fastdem_binary

try:
    EXE = fastdem_binary(os.environ.get("FASTDEM_EXE"))
except FileNotFoundError:
    EXE = None


def test_adapter_discovery_prefers_cache_and_preserves_existing_builds(tmp_path, monkeypatch):
    import robot_mapping

    monkeypatch.setattr(robot_mapping, "ROOT", tmp_path)
    relative = "build-mingw/fastdem_replay.exe" if os.name == "nt" else "build/fastdem_replay"
    legacy = tmp_path / "logs/fastdem" / relative
    cached = tmp_path / ".cache/fastdem" / relative
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert fastdem_binary() == legacy.resolve()
    cached.parent.mkdir(parents=True)
    cached.touch()
    assert fastdem_binary() == cached.resolve()
    assert fastdem_binary(legacy) == legacy.resolve()
    # An explicit typo must fail instead of silently selecting a different binary.
    with pytest.raises(FileNotFoundError, match="FastDEM not found"):
        fastdem_binary(tmp_path / "missing")


@pytest.mark.skipif(EXE is None, reason="Build the pinned FastDEM adapter first")
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
        [str(EXE), str(ROOT / "config/heightmaps/fastdem.yaml")],
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
