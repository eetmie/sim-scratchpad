"""The native RGB-D cameras must agree pixel for pixel with the independent geometric depth rays."""

import numpy as np
import pytest
import warp as wp
from lidar import TriangleScene, capture_depth, pinhole_rays
from sensor_contract import DEPTH_CAMERAS, DEPTH_SIZE
from soil import pose_matrix

pytestmark = pytest.mark.kitless


@pytest.mark.skipif(not wp.is_cuda_available(), reason="Requires CUDA")
def test_native_cameras_match_geometric_depth_through_soil_and_robot_changes():
    from rgb import CameraRig

    names = ["upper_carriage", "tilt_boom", "part"]
    poses = np.array([[0, 0, 0.5, 0, 0, 0, 1], [0.4, 0, 0.6, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]], np.float32)
    # A scalene triangle on its own link, 0.3 m in front of the cabin camera: any flip or swap moves its pixels.
    _, _, xyz, quat, _, _ = DEPTH_CAMERAS[0]
    poses[2, :3] = poses[0, :3] + xyz + 0.3 * pose_matrix(np.r_[np.zeros(3), quat])[:, 2]
    part = np.array([[0, 0, 0], [0.12, 0, 0], [0, 0.05, 0.03]], np.float32)
    faces = np.array([0, 1, 2], np.int32)
    wide = np.array([[-9, -9, 0], [9, -9, 0], [9, 9, 0], [-9, 9, 0]], np.float32)
    quad = np.array([0, 1, 2, 0, 2, 3], np.int32)
    scene = TriangleScene("cuda:0")
    scene.update("ground", wide + (0, 0, 0.02), quad)
    cameras = CameraRig(scene, [(2, part, faces, (0.02, 0.02, 0.02))], names, floor=-0.1, width=160, shadows=False)

    def compare(moving):
        scene.update("moving", *moving)
        scene.update("robot", part + poses[2, :3], faces)
        cameras.update(poses)
        seen = 0
        for name, parent, xyz, quat, fx, fy in DEPTH_CAMERAS:
            pose = poses[names.index(parent)]
            rays = pinhole_rays(fx * 160 / DEPTH_SIZE[0], fy * 120 / DEPTH_SIZE[1], 160, 120)
            rotation = pose_matrix(pose) @ pose_matrix(np.r_[np.zeros(3), quat])
            expected = capture_depth(scene, pose[:3] + pose_matrix(pose) @ np.asarray(xyz), rotation, rays, 160, 120)
            depth = cameras.depth_images[name]
            valid = np.isfinite(expected)
            seen += valid.sum()
            # Silhouette pixels may resolve to either surface; everything else must match.
            close = np.abs(depth[valid] - expected[valid]) < 2e-3
            assert close.mean() > 0.995, name
            assert cameras.rgb_images[name].shape == (120, 160, 3)
        return seen, cameras.depth_images["cabin"].copy(), cameras.rgb_images["cabin"].copy()

    seen, flat, rgb = compare((np.empty((0, 3)), np.empty(0, int)))
    assert seen > 5000
    sand = rgb[np.abs(flat - np.nanmax(flat)) < 0.2]  # far pixels are ground, not the dark robot part
    assert (sand[:, 0] > sand[:, 1]).all() and (sand[:, 1] > sand[:, 2]).all()
    # A raised patch with new topology must appear, then leave no ghost when the moving mesh empties.
    patch = (np.array([[0.2, -0.3, 0.1], [0.9, -0.3, 0.1], [0.9, 0.0, 0.1], [0.2, 0.0, 0.1]], np.float32), quad)
    _, raised, _ = compare(patch)
    assert np.nanmax(flat - raised) > 0.02
    _, emptied, _ = compare((np.empty((0, 3)), np.empty(0, int)))
    np.testing.assert_allclose(emptied, flat, atol=1e-5, equal_nan=True)
