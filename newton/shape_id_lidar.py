"""Newton BVH ray caster that also reports which collision shape each ray hit.

Newton's stock sensor already hits every shape (terrain, rocks, the robot's own links); this
adds the per-ray shape index so the caller can drop self-hits the way the RTX demo uses object
IDs: the ray still stops at the bucket (occlusion), its point is just not mapped.
Import after the simulation has launched.
"""

import newton
import numpy as np
import warp as wp
from isaaclab_newton.sensors.ray_caster.newton_raycast_sensor import NewtonRaycastSensor, _resolve_bvh_hits_kernel


class ShapeIdRaycastSensor(NewtonRaycastSensor):
    """NewtonRaycastSensor plus ``hit_shape_ids`` (one int per ray, -1 on a miss)."""

    def _initialize_impl(self):
        super()._initialize_impl()
        self._hit_shape = wp.full(self._num_envs * self.num_rays, -1, dtype=wp.int32, device=self._device)

    def _launch_raycast(self):
        # Same as the parent, with out_shape_id added; it stays inside the sensor's CUDA graph.
        self._update_ray_infos(self._is_outdated)
        newton.intersect_ray(
            self.backend.model,
            ray_origins=self._ray_starts_w_flat,
            ray_directions=self._ray_directions_w_flat,
            ray_worlds=self._ray_worlds,
            enable_global_world=True,
            out_dist=self._hit_dist,
            out_shape_id=self._hit_shape,
            out_normal=self._hit_normal,
        )
        wp.launch(
            _resolve_bvh_hits_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                self._is_outdated,
                self._ray_starts_w,
                self._ray_directions_w,
                self._hit_dist,
                self._hit_normal,
                float(self.cfg.max_distance),
                self.ray_cast_drift.warp,
            ],
            outputs=[self._data._ray_hits_w, self._data._ray_distances, self._data._ray_normals_w],
            device=self._device,
        )

    def scan(self):
        """Latest env-0 scan as NumPy: (ray starts [N, 3], unit directions [N, 3], distance or inf, shape ID or -1).

        Everything comes from the ray query's own buffers, which one intersect_ray call writes together.
        (The sensor's ``ray_hits_w`` is only rewritten for environments flagged as due, so it can lag the
        shape IDs by a scan, and ``data.pos_w`` is the carrier body without the mount offset.)
        """
        self.data  # refreshes the scan if it is due
        wp.synchronize_device(self._device)
        rays = slice(0, self.num_rays)
        starts = self._ray_starts_w_flat.numpy()[rays]
        directions = self._ray_directions_w_flat.numpy()[rays]
        distance = self._hit_dist.numpy()[rays]
        shape_ids = self._hit_shape.numpy()[rays]
        hit = (shape_ids >= 0) & (distance >= 0.0) & (distance <= self.cfg.max_distance)
        return starts, directions, np.where(hit, distance, np.inf), np.where(hit, shape_ids, -1)

    def shapes_under(self, prim_prefix: str) -> np.ndarray:
        """Boolean mask over all model shapes: True for shapes on bodies whose path starts with prim_prefix."""
        model = self.backend.model
        bodies = np.array([label.startswith(prim_prefix) for label in model.body_label] + [False])
        shape_body = model.shape_body.numpy()  # -1 (static) indexes the trailing False
        return bodies[shape_body]
