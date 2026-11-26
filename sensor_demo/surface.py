"""Opaque sand surface reconstructed from the dynamic particles, independently of physics."""

import numpy as np
import warp as wp
from newton.geometry import ParticleSurface
from sand_render import MeshPublisher, PointsPublisher
from soil import COLOR, HALF_WIDTH, PROFILE, pose_matrix, terrain_height, terrain_normal


@wp.kernel
def blend_ground_field(
    volume: wp.uint64,
    coordinates: wp.array(dtype=wp.vec3i),
    field: wp.array(dtype=float),
    height: wp.array2d(dtype=float),
    origin: float,
    cell: float,
    floor: float,
    threshold: float,
    scale: float,
    blend: float,
    voxel: float,
):
    i = wp.tid()
    if i >= wp.volume_voxel_count(volume):
        return
    ijk = coordinates[i]
    # Newton's extraction nodes lie on the integer lattice; the NanoVDB
    # cell-center transform would introduce a half-voxel height mismatch.
    p = voxel * wp.vec3(float(ijk[0]), float(ijk[1]), float(ijk[2]))
    ground = terrain_height(height, p, origin, cell, floor)
    # Smooth union in signed-distance units, converted back to the extractor's
    # density convention. This joins the two representations before meshing.
    a = scale * wp.log(threshold / wp.max(field[i], 1e-6))
    b = p[2] - ground
    overlap = wp.max(blend - wp.abs(a - b), 0.0)
    distance = wp.min(a, b) - overlap * overlap / (4.0 * blend)
    field[i] = threshold - distance * threshold / scale


@wp.kernel
def classify_spray(
    volume: wp.uint64,
    field: wp.array(dtype=float),
    x: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=int),
    voxel: float,
    mask: wp.array(dtype=int),
):
    i = wp.tid()
    mask[i] = 0
    if active[i] != 0:
        density = wp.volume_sample_index(volume, x[i] / voxel, wp.Volume.LINEAR, field, 0.0)
        if density < 0.55:
            mask[i] = 1


@wp.kernel
def match_ground_normals(
    vertices: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=int),
    height: wp.array2d(dtype=float),
    origin: float,
    cell: float,
    floor: float,
    blend: float,
):
    i = wp.tid()
    if i >= offsets[1]:
        return
    p = vertices[i]
    ground = terrain_height(height, p, origin, cell, floor)
    distance = p[2] - ground
    if distance >= 0.0 and distance < blend:
        t = wp.clamp(distance / blend, 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        normal = terrain_normal(height, p, origin, cell, floor)
        normals[i] = wp.normalize((1.0 - t) * normal + t * normals[i])


@wp.kernel
def hide_redundant_ground(
    vertices: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    height: wp.array2d(dtype=float),
    origin: float,
    cell: float,
    floor: float,
):
    triangle = wp.tid()
    if 3 * triangle >= offsets[1]:
        return
    redundant = int(1)
    for k in range(3):
        p = vertices[indices[3 * triangle + k]]
        h = terrain_height(height, p, origin, cell, floor)
        if p[2] > h + 0.00015:
            redundant = 0
    if redundant != 0:
        # The height-grid mesh owns this same surface. Removing duplicate
        # triangles prevents a rectangular patch edge and depth flicker.
        indices[3 * triangle + 1] = indices[3 * triangle]
        indices[3 * triangle + 2] = indices[3 * triangle]


class SandSurface:
    """Use Newton's fluid-style surface reconstruction with a rough sand material."""

    def __init__(self, visualizers, soil, voxel_size=None, blend_width=None):
        self.soil = soil
        self.blend_width = blend_width or 1.5 * soil.cfg.cell
        self.height_wp = wp.empty_like(soil.height_wp)
        self.coordinates = wp.empty(262144, dtype=wp.vec3i, device=soil.device)
        self.radii_wp = wp.empty_like(soil.radius_wp)
        self.render_flags = wp.zeros_like(soil.active_wp)
        self.display_height = soil.render_height()
        self.ground_changed = True
        self.spray_mask = wp.zeros(soil.cfg.capacity, dtype=int, device=soil.device)
        self.spray_shown = np.zeros(soil.cfg.capacity, dtype=bool)
        self.spray = PointsPublisher(visualizers, "/World/SoilSpray", radius=0.2 * soil.cfg.cell)
        self.publisher = MeshPublisher(
            visualizers, "/World/MovingSoil", color=tuple(float(c) for c in COLOR), roughness=0.95
        )
        self.surface = ParticleSurface(
            voxel_size=voxel_size or 0.5 * soil.cfg.cell,
            kernel_radius=2.2 * soil.cfg.cell,
            threshold=0.5,
            smooth_lambda=0.35,
            anisotropic=False,
            field_smooth_iterations=1,
            mesh_smooth_iterations=0,
            max_grid_cells=262144,
            device=soil.device,
        )
        self.graph = None
        self.mesh = None
        self.triangles = 0

    def _extract(self):
        field = self.surface.update_field(self.soil.x, self.radii_wp, particle_flags=self.render_flags)
        field.volume.get_voxels(out=self.coordinates)
        wp.launch(
            blend_ground_field,
            len(field.voxel_data),
            [
                field.volume.id,
                self.coordinates,
                field.voxel_data,
                self.height_wp,
                self.soil.origin,
                self.soil.cfg.cell,
                self.soil.cfg.floor,
                0.5,
                1.1 * self.soil.cfg.cell,
                self.blend_width,
                self.surface.voxel_size,
            ],
            device=self.soil.device,
        )
        mesh = self.surface.resurface()
        wp.launch(
            classify_spray,
            self.soil.cfg.capacity,
            [
                field.volume.id,
                field.voxel_data,
                self.soil.x,
                self.render_flags,
                self.surface.voxel_size,
                self.spray_mask,
            ],
            device=self.soil.device,
        )
        wp.launch(
            match_ground_normals,
            len(mesh.vertices),
            [
                mesh.vertices,
                mesh.normals,
                mesh.vertex_world_offsets,
                self.height_wp,
                self.soil.origin,
                self.soil.cfg.cell,
                self.soil.cfg.floor,
                self.blend_width,
            ],
            device=self.soil.device,
        )
        wp.launch(
            hide_redundant_ground,
            len(mesh.indices) // 3,
            [
                mesh.vertices,
                mesh.indices,
                mesh.index_world_offsets,
                self.height_wp,
                self.soil.origin,
                self.soil.cfg.cell,
                self.soil.cfg.floor,
            ],
            device=self.soil.device,
        )
        return mesh

    def _prepare(self):
        """Split near-ground render volume continuously between the two representations."""
        soil = self.soil
        ids = np.flatnonzero(soil.active)
        p = soil.positions[ids]
        cell = soil.cfg.cell
        ix = np.floor((p[:, 0] - soil.origin) / cell).astype(int)
        iy = np.floor((p[:, 1] - soil.origin) / cell).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < soil.n) & (iy < soil.n)
        ground = np.full(len(ids), soil.cfg.floor, dtype=np.float32)
        ground[inside] = soil.height[iy[inside], ix[inside]]
        gap = p[:, 2] - soil.radii[ids] - ground
        t = np.clip((gap - 0.5 * cell) / (2.5 * cell), 0, 1)
        near_ground = 1 - t * t * (3 - 2 * t)
        t = np.clip((np.abs(soil.velocities[ids, 2]) - 0.1) / 0.3, 0, 1)
        grounded = near_ground * (1 - t * t * (3 - 2 * t)) * inside
        if soil.previous_pose is not None:
            pose = soil.previous_pose
            local = (p - pose[:3]) @ pose_matrix(pose)
            cavity = np.ones(len(ids), dtype=bool)
            for a, b in zip(PROFILE, np.roll(PROFILE, -1, axis=0)):
                edge = b - a
                rel = local[:, 1:] - a
                cavity &= edge[0] * rel[:, 1] - edge[1] * rel[:, 0] >= 0
            cavity &= np.abs(local[:, 0]) < HALF_WIDTH
            grounded[cavity] = 0  # carried material stays with the bucket
        height = soil.height.copy()
        deposit = np.zeros_like(height)
        ceiling = height.copy()
        np.add.at(deposit, (iy[inside], ix[inside]), soil.volumes[ids[inside]] * grounded[inside] / cell**2)
        contributes = inside & (grounded > 1e-4)
        np.maximum.at(ceiling, (iy[contributes], ix[contributes]), p[contributes, 2] + soil.radii[ids[contributes]])
        # Many compressed grains can occupy one XY cell during lifting.
        # Never turn their summed volume into a ground spire above the actual
        # particle envelope; the remaining volume stays in the 3D surface.
        available = np.maximum(ceiling - height, 0)
        scale = np.minimum(1, available / np.maximum(deposit, 1e-12))
        grounded[inside] *= scale[iy[inside], ix[inside]]
        height += deposit * scale
        self.grounded_fraction = np.zeros(soil.cfg.capacity, dtype=np.float32)
        self.grounded_fraction[ids] = grounded
        height = soil.render_height(height)
        self.ground_changed = not np.array_equal(self.display_height, height)
        if self.ground_changed:  # consumers compare identity to skip rebuilding an unchanged ground
            self.display_height = height
        self.height_wp.assign(height)
        volume = np.zeros(soil.cfg.capacity, dtype=np.float32)
        volume[ids] = soil.volumes[ids] * (1 - grounded)
        self.radii_wp.assign((0.5 * np.cbrt(volume)).astype(np.float32))
        flags = np.zeros(soil.cfg.capacity, dtype=np.int32)
        flags[ids] = (1 - grounded) > 1e-3
        self.render_flags.assign(flags)

    def update(self):
        """Stage the current dynamic surface and return its triangle count."""
        self._prepare()
        if not self.soil.active.any():
            self.publisher.update(None, None)
            self.spray.update(np.empty((0, 3), np.float32), np.empty((0, 3), np.float32))
            self.triangles = 0
            return 0
        if self.graph is None:
            self.mesh = self._extract()
            if wp.get_device(self.soil.device).is_cuda:
                with wp.ScopedCapture(device=self.soil.device) as capture:
                    self.mesh = self._extract()
                self.graph = capture.graph
                wp.capture_launch(self.graph)
        else:
            wp.capture_launch(self.graph)
        vertices, indices, normals = self.mesh.to_arrays()
        # Only falling/thrown grains are spray. Slow sparse grains (resting, or riding the bucket)
        # sample low density too, but drawing them as beads makes them flicker and look out of place.
        speed = np.linalg.norm(self.soil.velocities, axis=1)
        mask = (self.spray_mask.numpy() != 0) & (self.soil.active != 0)
        mask &= (self.grounded_fraction < 0.1) & (speed > 0.25)
        self.spray_shown = mask
        points = self.soil.positions[mask]
        self.spray.update(points, np.tile(COLOR, (len(points), 1)))
        if vertices is None:
            self.publisher.update(None, None)
            self.triangles = 0
            return 0
        self.triangles = len(indices) // 3
        if self.publisher.kit is not None:
            self.publisher.update(vertices.numpy(), indices.numpy(), normals.numpy())
        else:
            for visualizer in self.publisher.gl:
                visualizer.log_mesh(
                    self.publisher.path,
                    vertices,
                    indices,
                    normals=normals,
                    color=tuple(float(c) for c in COLOR),
                    roughness=0.95,
                    metallic=0.0,
                    opacity=1.0,
                    dynamic=True,
                    backface_culling=False,
                )
        return self.triangles
