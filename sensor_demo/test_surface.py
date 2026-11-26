"""Surface reconstruction must honor the active pool and leave physics untouched."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp
from soil import Soil, SoilConfig

pytestmark = pytest.mark.kitless


def test_surface_excludes_parked_particles_and_preserves_physics(monkeypatch):
    if not wp.is_cuda_available():
        pytest.skip("Newton sparse surface extraction requires CUDA")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent / "newton"))
    from surface import SandSurface

    soil = Soil(SoilConfig(capacity=2000), device="cuda:0")
    soil.step([0.5, 0, 0.10, 0, 0, 0, 1], 1 / 60)
    # Parked slots can retain positive radii after a previous particle settles.
    soil.radii[soil.active == 0] = 0.535 * soil.cfg.cell
    soil.volumes[soil.active == 0] = soil.cfg.cell**3
    soil.radius_wp.assign(soil.radii)
    # A known flat ground plane makes continuity at the outer collar measurable.
    soil.height.fill(0.015)
    soil.positions[soil.active != 0, 2] += 0.025
    soil._upload_particles()
    before_x, before_v = soil.x.numpy(), soil.v.numpy()
    before_height = soil.height.copy()
    surface = SandSurface([], soil)
    assert surface.update() > 0
    assert surface.update() > 0  # also exercise captured extraction replay
    vertices, _, normals = surface.mesh.to_arrays()
    points = vertices.numpy()
    active = soil.positions[soil.active != 0]
    # The local reconstructed ground collar may extend beyond the grains;
    # parked slots at x/y=10 m must never become visible geometry.
    assert np.all(points.min(axis=0) >= active.min(axis=0) - 0.12)
    assert np.all(points.max(axis=0) <= active.max(axis=0) + 0.12)
    ns = normals.numpy()
    assert np.isfinite(ns).all()
    outside = (points[:, 0] < active[:, 0].min() - 0.025) | (points[:, 0] > active[:, 0].max() + 0.025)
    collar = outside & (np.abs(points[:, 2] - 0.015) < 0.0005)
    assert collar.any()
    assert np.min(ns[collar, 2]) > 0.98
    np.testing.assert_array_equal(soil.x.numpy(), before_x)
    np.testing.assert_array_equal(soil.v.numpy(), before_v)
    np.testing.assert_array_equal(soil.height, before_height)
    # A falling cloud must remain airborne in the view instead of raising terrain below it.
    soil.positions[soil.active != 0, 2] += 0.4
    soil.velocities[soil.active != 0, 2] = -1.0
    soil._upload_particles()
    surface.update()
    np.testing.assert_allclose(surface.display_height, 0.015, atol=1e-6)


def test_compressed_grounded_grains_do_not_become_a_render_spire(monkeypatch):
    if not wp.is_cuda_available():
        pytest.skip("Newton sparse surface extraction requires CUDA")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent / "newton"))
    from surface import SandSurface

    soil = Soil(SoilConfig(size=0.2, capacity=32), device="cuda:0")
    soil.height.fill(0)
    soil.active[:21] = 1
    soil.positions[:20] = np.column_stack((np.zeros(20), np.zeros(20), np.linspace(0.005, 0.025, 20)))
    soil.radii[:20] = 0.005
    soil.volumes[:20] = 1e-6
    # A carried grain above the same column must not authorize a tall ground mound.
    soil.positions[20] = [0, 0, 0.3]
    soil.radii[20] = 0.005
    soil.volumes[20] = 1e-6
    soil._upload_particles()
    surface = SandSurface([], soil)
    surface.update()
    assert surface.display_height.max() <= 0.030001
    assert np.any(surface.grounded_fraction[:20] < 1)  # excess remains three-dimensional
