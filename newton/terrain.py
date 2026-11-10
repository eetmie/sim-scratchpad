"""Crop Isaac's rough_plane terrain (96 m, 1.3 M triangles) to a small patch around the excavator.

demo_env.usd references the full terrain at ROUGH_PLANE_OFFSET; the patch keeps that placement,
so the Newton scene sees the same bumps as the RTX one. Newton gets a ~2k-triangle static mesh
instead of the whole plane. The patch is cached next to this file.
"""

import os

import numpy as np

ROUGH_PLANE_PATH = "Environments/Terrains/rough_plane.usd"  # under ISAAC_NUCLEUS_DIR
ROUGH_PLANE_OFFSET = (-0.4384718436746712, 0.9400099954447185, -0.07993896840716719)  # as in demo_env.usd
PATCH_X_M = (-1.2, 2.0)
PATCH_Y_M = (-1.4, 1.4)
PATCH_COLOR = (0.55, 0.47, 0.36)  # sandy brown
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "rough_plane_patch.usda")


def build_terrain_patch(cache_path=CACHE_PATH):
    """Return a USD file holding the cropped, collidable terrain patch; builds it on first use."""
    if os.path.isfile(cache_path):
        return cache_path
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

    print("[Terrain] Downloading rough_plane (~170 MB, once) and cropping a patch...")
    source = Usd.Stage.Open(retrieve_file_path(f"{ISAAC_NUCLEUS_DIR}/{ROUGH_PLANE_PATH}"))
    mesh_prim = next(p for p in source.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh = UsdGeom.Mesh(mesh_prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not (counts == 3).all():
        raise ValueError("Expected a triangulated rough_plane mesh.")
    to_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim))  # row-vector matrix
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    points = points @ to_world[:3, :3] + to_world[3, :3] + np.asarray(ROUGH_PLANE_OFFSET)
    triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)

    corners = points[triangles]  # (T, 3, 3)
    keep = (
        (corners[..., 0] >= PATCH_X_M[0]).all(1)
        & (corners[..., 0] <= PATCH_X_M[1]).all(1)
        & (corners[..., 1] >= PATCH_Y_M[0]).all(1)
        & (corners[..., 1] <= PATCH_Y_M[1]).all(1)
    )
    used, triangles = np.unique(triangles[keep], return_inverse=True)
    points = points[used].astype(np.float32)
    triangles = triangles.reshape(-1, 3)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    stage = Usd.Stage.CreateNew(cache_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Terrain")
    stage.SetDefaultPrim(root.GetPrim())
    patch = UsdGeom.Mesh.Define(stage, "/Terrain/rough_patch")
    patch.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    patch.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(triangles)))
    patch.CreateFaceVertexIndicesAttr(Vt.IntArray(triangles.ravel().tolist()))
    patch.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    patch.CreateDisplayColorAttr([Gf.Vec3f(*PATCH_COLOR)])
    patch.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(patch.GetPointsAttr().Get()))
    UsdPhysics.CollisionAPI.Apply(patch.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(patch.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    stage.GetRootLayer().Save()
    z = points[:, 2]
    print(f"[Terrain] Patch: {len(triangles)} triangles, z {z.min():+.3f}..{z.max():+.3f} m -> {cache_path}")
    return cache_path
