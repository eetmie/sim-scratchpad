# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent ray geometry, moving occlusion, range and cabin-frame image contracts."""

import numpy as np
import pytest
from lidar import TriangleScene, capture_depth, map_pixels, pinhole_rays, rasterize

pytestmark = pytest.mark.kitless


def plane(z):
    return np.array([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]], np.float32), np.array(
        [0, 1, 2, 0, 2, 3], np.int32
    )


def test_nearest_surface_robot_occlusion_and_moving_mesh_updates():
    scene = TriangleScene("cpu")
    scene.update("ground", *plane(0))
    scene.update("moving", *plane(0.3))
    scene.update("robot", *plane(0.6))
    origins = np.array([[0.2, 0.1, 1]], np.float32)
    directions = np.array([[0, 0, -1]], np.float32)
    distance, label = scene.cast(origins, directions, far=2)
    np.testing.assert_allclose(distance, 0.4, atol=1e-6)
    assert label[0] == 3  # self hit stops the ray; deleting it cannot reveal hidden soil
    distance, label = scene.cast(origins, directions, far=2, include_robot=False)
    np.testing.assert_allclose(distance, 0.7, atol=1e-6)
    assert label[0] == 2  # lifted soil is not collapsed onto the ground plane
    scene.update("moving", *plane(0.8))  # same topology, BVH must refit
    distance, label = scene.cast(origins, directions, far=2)
    np.testing.assert_allclose(distance, 0.2, atol=1e-6)
    assert label[0] == 2
    _, label = scene.cast(origins, directions, far=0.1)
    assert label[0] == 0
    scene.update("moving", np.empty((0, 3)), np.empty(0, int))
    distance, label = scene.cast(origins, directions, far=2, include_robot=False)
    np.testing.assert_allclose(distance, 1, atol=1e-6)
    assert label[0] == 1  # emptied dynamic mesh cannot leave ghost returns


def test_cabin_frame_latest_scan_orientation_height_and_unknown_cells():
    q = np.sqrt(0.5)
    cabin = np.array([1, 2, 0.4, 0, 0, q, q])  # cabin yaw +90 degrees
    points = np.array([[0.89375, 2.30625, 0.25], [0.89375, 2.30625, 0.2], [20, 20, 20]])
    height = rasterize(points, cabin)
    assert np.isfinite(height).sum() == 1
    assert height[47, 23] == pytest.approx(-0.15)
    pixels = map_pixels(height)
    assert pixels[47, 23] == 128
    assert np.count_nonzero(pixels) == 1
    # A new empty scan must not retain stale measurements from the last view.
    assert np.isnan(rasterize(np.empty((0, 3)), cabin)).all()


def test_geometric_camera_reports_optical_depth_and_masks_misses():
    scene = TriangleScene("cpu")
    scene.update("ground", *plane(1))
    rays = pinhole_rays(2, 2, 4, 4)
    depth = capture_depth(scene, np.zeros(3), np.eye(3), rays, 4, 4)
    np.testing.assert_allclose(depth, 1, atol=1e-6)  # corner rays travel farther, but plane Z is constant
    depth = capture_depth(scene, np.zeros(3), np.diag([1, -1, -1]), rays, 4, 4)
    assert np.isnan(depth).all()
