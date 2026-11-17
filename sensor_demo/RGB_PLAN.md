# RGB cameras: what was built and what is left

`rgb.py` implements the native option of the original plan: `newton.sensors.SensorCamera` on a render-only
model. The fallback hand-written shader was not needed. Measurements are in [runs/RESULTS.md](runs/RESULTS.md).

## How it works

- The render model holds one body per robot link with the USD visual meshes (one shape per material), the
  floor slab, the support block, and two soil shapes. It has no collision and is never stepped.
- Each camera tick copies the link poses into `state.body_q` and refits the shape BVH.
- The soil shapes do not own geometry. Before each render their `model.shape_source_ptr` entries are pointed at
  the lidar scene's current `wp.Mesh` for the ground and the moving surface, so a topology change costs one
  mesh build that the lidar needed anyway.
- Each camera renders with its own ray bundle, because the two calibrated focal lengths differ.

## What was tried and rejected

- **Fixed-capacity triangle buffer with BVH refit** for the moving surface: refitting a tree built over
  degenerate triangles made two 640x480 views take seconds. Swapping in a rebuilt mesh takes about 1 ms.
- **Rendering Isaac Lab's simulation model**: it only holds the invisible collision meshes, and soil cannot be
  added to it after it is finalized.

## Things that depend on Newton internals

Installed Newton is 1.9.0.dev0. Recheck these after upgrading it; `test_rgb.py` fails if either breaks.

- Writing `model.shape_source_ptr` to swap a shape's mesh is not a documented API.
- Newton caches each shape's local bounds in `model.bvh_shape_bounds` at finalize. The soil shapes' bounds are
  widened by hand so swapped-in meshes are not culled.

## Left to do

- The arm camera mount is still the placeholder in `sensor_contract.py`.
- The principal point is the image centre; the real ones are about 5 px off.
- RGB uses the depth (IR1) frame and intrinsics. The real colour sensor has its own lens and a small offset.
- The ground renders with flat triangle normals, which shows as a faint grid on slopes.
- Shading is one light, shadows and flat material colours, with no textures or noise. Before training a vision
  policy on it, compare with real cabin and arm recordings and add lighting and colour variation.
- Native depth is not written into `sensors/scan_*.npz`; those still hold the 160x120 geometric depth.
