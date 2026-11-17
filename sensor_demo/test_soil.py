# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical checks without Kit, the robot asset, or a display.

SOIL_TEST_DEVICE=cuda:0 selects the GPU; CPU is the default.
"""

import os

import numpy as np
import pytest
from soil import Soil, SoilConfig

pytestmark = pytest.mark.kitless
DEVICE = os.environ.get("SOIL_TEST_DEVICE", "cpu")


def test_pool_exhaustion_conserves_volume_and_reset_restores_block():
    """A full particle pool must leave unconverted ground intact, including partial layers."""
    soil = Soil(SoilConfig(cell=0.02, capacity=32, substeps=2, iterations=3), device=DEVICE)
    initial_volume = soil.volume()
    initial_height = soil.height.copy()
    for z in (0.01, -0.02, -0.05):
        soil.step([0.5, 0, z, 0, 0, 0, 1], 1 / 60)
    assert soil.saturated
    assert 0 < soil.active.sum() <= 32
    assert soil.volume() == pytest.approx(initial_volume, abs=1e-7)
    assert soil.height.min() >= soil.cfg.floor - 1e-6
    soil.reset()
    assert not soil.active.any()
    np.testing.assert_array_equal(soil.height, initial_height)


def test_lift_tip_and_settle_preserve_volume_without_ejecting_soil():
    """A moving open bucket carries soil upward, then releases it to settle on supported ground."""
    soil = Soil(SoilConfig(cell=0.02, capacity=1000, substeps=6, iterations=5), device=DEVICE)
    pose = np.array([0.5, 0, 0.01, 0, 0, 0, 1], np.float32)
    soil.step(pose, 1 / 60)
    activated = soil.active.sum()
    assert activated > 0
    carried = False
    for step in range(300):
        pose[2] = 0.01 + min(step / 45, 1) * 0.45
        angle = np.pi * np.clip((step - 60) / 45, 0, 1)
        pose[3], pose[6] = np.sin(angle / 2), np.cos(angle / 2)
        soil.step(pose, 1 / 60)
        moving = soil.positions[soil.active != 0]
        assert np.isfinite(moving).all()
        assert moving[:, 2].max(initial=0) < 0.85
        assert moving[:, 2].min(initial=0) >= soil.cfg.floor - 0.02
        if step == 59:
            carried = np.any(moving[:, 2] > 0.25)
    assert carried
    assert soil.active.sum() < activated * 0.25
    assert soil.volume() == pytest.approx(soil.initial_volume, abs=1e-7)


def test_stationary_bucket_and_cut_edges_do_not_launch_particles():
    """Overlap recovery at a cut wall must not become an upward velocity impulse."""
    soil = Soil(SoilConfig(capacity=2000), device=DEVICE)
    pose = np.array([0.5, 0, 0.01, 0, 0, 0, 1], np.float32)
    late_upward_speed = 0.0
    for step in range(180):
        pose[2] = 0.01 + min(step / 45, 1) * 0.45
        soil.step(pose, 1 / 60)
        if step >= 120:
            late_upward_speed = max(late_upward_speed, soil.velocities[soil.active != 0, 2].max(initial=0))
    assert late_upward_speed < 0.75


def test_initial_overlap_recovery_does_not_add_kinetic_energy():
    """With a stationary tool, initial placement may relax but cannot power fast grains."""
    soil = Soil(SoilConfig(capacity=2000), device=DEVICE)
    soil.step([0.5, 0, 0.01, 0, 0, 0, 1], 1 / 60)
    speed = np.linalg.norm(soil.velocities[soil.active != 0], axis=1)
    # All parcels start at rest: free fall for this interval is the only source of speed.
    assert speed.max(initial=0) <= 9.81 / 60 * 1.01


@pytest.mark.parametrize(
    "start, velocity, axis, limit, side",
    [
        ((0, 0.05, -0.06), (0, 0, -2), 2, -0.086, 1),
        ((0.03, 0.05, -0.06), (2, 0, 0), 0, 0.05, -1),
    ],
)
def test_swept_bucket_contact_stops_floor_and_side_tunneling(start, velocity, axis, limit, side):
    """Cross a whole plate between substeps, ending outside the point-query band."""
    soil = Soil(SoilConfig(size=0.2, capacity=1, substeps=1, iterations=2), device=DEVICE)
    soil.positions[0] = np.asarray(start) + (0, 0, 0.3)
    soil.velocities[0] = velocity
    soil.active[0] = 1
    soil.radii[0] = 0.002
    soil.volumes[0] = (0.002 / 0.535) ** 3
    soil._upload_particles()
    soil.step([0, 0, 0.3, 0, 0, 0, 1], 1 / 30)
    local = soil.positions[0] - (0, 0, 0.3)
    clearance = soil.radii[0] + soil.cfg.bucket_margin
    assert side * (local[axis] - limit) >= clearance - 1e-5
