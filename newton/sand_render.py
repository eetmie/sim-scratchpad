"""Draw the sand bed and the bucket load in Kit (a USD mesh) and/or the Newton viewer (log_mesh)."""

import numpy as np
import warp as wp
from sand import SAND_COLOR


class MeshPublisher:
    """One dynamic triangle mesh, pushed to every active Kit / Newton GL visualizer."""

    def __init__(self, visualizers, path, color=SAND_COLOR, roughness=0.95):
        self.path, self.color, self.roughness = path, color, roughness
        self.gl = [v for v in visualizers if hasattr(v, "log_mesh") and type(v).__name__.startswith("Newton")]
        self.kit = None
        if any(type(v).__name__ == "KitVisualizer" for v in visualizers):
            from pxr import UsdGeom

            from isaaclab.sim import PreviewSurfaceCfg, bind_visual_material, get_current_stage, spawn_preview_surface

            self.kit = UsdGeom.Mesh.Define(get_current_stage(), path)
            self.kit.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            self.kit.CreateDoubleSidedAttr(True)
            spawn_preview_surface(f"{path}/material", PreviewSurfaceCfg(diffuse_color=color, roughness=roughness))
            bind_visual_material(path, f"{path}/material")
        self._indices = None

    def update(self, points, indices, normals=None):
        """Show the mesh; ``points=None`` hides it."""
        hidden = points is None or not len(points)
        if hidden:
            points = np.zeros((3, 3), np.float32)
            indices = np.array([0, 1, 2], np.int32)
        if self.kit is not None:
            from pxr import UsdGeom, Vt

            vt_points = Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32))
            self.kit.GetPointsAttr().Set(vt_points)
            if (
                self._indices is None
                or len(self._indices) != len(indices)
                or not np.array_equal(self._indices, indices)
            ):
                self.kit.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices.astype(np.int32)))
                self.kit.GetFaceVertexCountsAttr().Set(Vt.IntArray(len(indices) // 3, 3))
                self._indices = indices.copy()
            if normals is not None and not hidden:
                self.kit.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                self.kit.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, dtype=np.float32)))
            self.kit.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(vt_points))
            self.kit.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible if hidden else UsdGeom.Tokens.inherited)
        for viewer in self.gl:
            device = "cuda:0" if wp.is_cuda_available() else "cpu"
            viewer.log_mesh(
                self.path,
                wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
                wp.array(indices.astype(np.int32), dtype=wp.int32, device=device),
                normals=None if normals is None or hidden else wp.array(normals, dtype=wp.vec3, device=device),
                hidden=hidden,
                backface_culling=False,
                color=self.color,
                roughness=self.roughness,
                metallic=0.0,
                dynamic=True,
            )


class PointsPublisher:
    """A dynamic point cloud (sand grains) with per-point colours, for Kit and/or the Newton viewer."""

    def __init__(self, visualizers, path, radius):
        self.path, self.radius = path, radius
        self.kit = None
        self._staged = None
        # The Newton visualizer stages meshes for its next frame but not points, and the GL viewer only takes
        # geometry between begin_frame() and end_frame(). So log the staged points just before end_frame().
        for visualizer in visualizers:
            viewer = getattr(visualizer, "_viewer", None)
            if type(visualizer).__name__ == "NewtonGLVisualizer" and viewer is not None:
                end_frame = viewer.end_frame

                def end_frame_with_points(viewer=viewer, end_frame=end_frame):
                    if self._staged is not None:
                        viewer.log_points(self.path, *self._staged)
                    end_frame()

                viewer.end_frame = end_frame_with_points
        if any(type(v).__name__ == "KitVisualizer" for v in visualizers):
            from pxr import UsdGeom

            from isaaclab.sim import PreviewSurfaceCfg, bind_visual_material, get_current_stage, spawn_preview_surface

            self.kit = UsdGeom.Points.Define(get_current_stage(), path)
            self.kit.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
            spawn_preview_surface(f"{path}/material", PreviewSurfaceCfg(diffuse_color=SAND_COLOR, roughness=0.95))
            bind_visual_material(path, f"{path}/material")
            self._count = -1

    def update(self, points, colors):
        hidden = not len(points)
        if self.kit is not None:
            from pxr import UsdGeom, Vt

            if hidden:
                self.kit.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            else:
                vt_points = Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32))
                self.kit.GetPointsAttr().Set(vt_points)
                if len(points) != self._count:
                    self.kit.GetWidthsAttr().Set(Vt.FloatArray(len(points), 2 * self.radius))
                    self.kit.GetDisplayColorAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colors, np.float32)))
                    self._count = len(points)
                lo, hi = points.min(axis=0) - self.radius, points.max(axis=0) + self.radius
                self.kit.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([lo, hi]).astype(np.float32)))
                self.kit.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        if hidden:
            self._staged = (None,)
        else:
            self._staged = (
                wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
                self.radius,
                wp.array(colors.astype(np.float32), dtype=wp.vec3, device=device),
            )
