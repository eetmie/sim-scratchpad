# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cabin and arm RGB-D cameras: Newton's Warp ray tracer on a render-only model. No Kit, no ROS."""

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorCamera
from sensor_contract import DEPTH_CAMERAS, DEPTH_SIZE
from soil import COLOR

SKY = 0xFFE6D2B4  # packed ABGR
OPTICAL_TO_GL = wp.quat(1.0, 0.0, 0.0, 0.0)  # contract frames look along +Z with y down; Newton cameras along -Z, y up


def robot_visuals(usd_path, body_names):
    """Each link's visual triangles per material, in link coordinates: (body, points, faces, sRGB colour).

    The exported URDF meshes lost the asset's materials, so the cameras read the USD the viewer shows.
    """
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(usd_path))
    parts = []
    for body, name in enumerate(body_names):
        link = stage.GetDefaultPrim().GetChild(name)
        mesh = UsdGeom.Mesh(link.GetChild("visuals").GetChild("Mesh"))
        to_link = np.array(
            UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(0)
            * UsdGeom.Xformable(link).ComputeLocalToWorldTransform(0).GetInverse()
        )
        points = (np.array(mesh.GetPointsAttr().Get()) @ to_link[:3, :3] + to_link[3, :3]).astype(np.float32)
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3)  # the asset is triangulated
        for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
            material = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
            color = next(
                c.Get()
                for c in (UsdShade.Shader(p).GetInput("diffuseColor") for p in Usd.PrimRange(material.GetPrim()))
                if c
            )
            parts.append((body, points, faces[np.array(subset.GetIndicesAttr().Get())].ravel(), tuple(color)))
    return parts


class CameraRig:
    """Renders the lidar rig's soil meshes and the robot links from the contract's camera mounts."""

    def __init__(
        self, scene, robot_parts, body_names, floor, width=DEPTH_SIZE[0], shadows=True, cameras=None, props=()
    ):
        """``cameras``: names from the contract, all by default. ``props``: movable (points, faces, colour) meshes."""
        self.scene, self.names = scene, list(body_names)
        self.cameras = [c for c in DEPTH_CAMERAS if cameras is None or c[0] in cameras]
        self.prop_poses = np.tile(np.array([0, 0, 0, 0, 0, 0, 1], np.float32), (len(props), 1))  # xyz + xyzw
        self.width, self.height = width, round(width * DEPTH_SIZE[1] / DEPTH_SIZE[0])
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
        for _ in self.names:
            builder.add_body()
        for body, points, faces, color in robot_parts:
            builder.add_shape_mesh(
                body=body, mesh=newton.Mesh(points, faces, compute_inertia=False), cfg=cfg, color=color
            )
        self.prop_shapes = [
            builder.add_shape_mesh(
                body=builder.add_body(), mesh=newton.Mesh(points, faces, compute_inertia=False), cfg=cfg, color=color
            )
            for points, faces, color in props
        ]
        # Same support block and floor slab as the viewer scene (demo.py SceneCfg).
        builder.add_shape_box(
            -1,
            xform=wp.transform((0, 0, floor / 2), wp.quat_identity()),
            hx=0.18,
            hy=0.15,
            hz=-floor / 2,
            cfg=cfg,
            color=(0.22, 0.23, 0.24),
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform((0, 0, floor - 0.026), wp.quat_identity()),
            hx=4,
            hy=4,
            hz=0.025,
            cfg=cfg,
            color=(0.19, 0.20, 0.21),
        )
        # Soil shapes start as a hidden placeholder; update() points them at the lidar scene's current meshes.
        hidden = np.array([[0, 0, -9], [1e-3, 0, -9], [0, 1e-3, -9]], np.float32)
        self.soil_shapes = {
            name: builder.add_shape_mesh(
                body=-1, mesh=newton.Mesh(hidden, [0, 1, 2], compute_inertia=False), cfg=cfg, color=tuple(COLOR)
            )
            for name in ("ground", "moving")
        }
        self.model = builder.finalize(scene.device)
        self.state = self.model.state()
        self.pointers = self.model.shape_source_ptr.numpy()
        self.hidden = self.pointers[self.soil_shapes["moving"]]
        # Newton caches each shape's local bounds at finalize; widen the soil's so swapped-in meshes are not culled.
        bounds = self.model.bvh_shape_bounds.numpy()
        for shape in self.soil_shapes.values():
            bounds[shape] = ((-4, -4, floor - 1), (4, 4, 4))
        self.model.bvh_shape_bounds.assign(bounds)
        self.sensor = SensorCamera(
            self.model,
            default_clear_data=SensorCamera.ClearData(clear_color=SKY),
            # Soil and bucket shells are open surfaces, so both sides must render.
            default_render_config=SensorCamera.RenderConfig(enable_shadows=shadows, enable_backface_culling=False),
        )
        self.sensor.create_default_light(enable_shadows=shadows)
        scale = width / DEPTH_SIZE[0]
        # shortcut: centred principal point like sensor_contract; pass the calibrated cx, cy here when it gains them.
        self.rays = {
            name: SensorCamera.compute_camera_rays_pinhole_opencv(
                width, self.height, fx * scale, fy * scale, width / 2, self.height / 2, device=scene.device
            )
            for name, _, _, _, fx, fy in self.cameras
        }
        self.transform = wp.empty(1, dtype=wp.transformf, device=scene.device)
        self.color = self.sensor.allocate_image_color(1, width, self.height)
        self.depth = self.sensor.allocate_image_depth_forward(1, width, self.height)
        self.rgb_images, self.depth_images, self.count = {}, {}, 0

    def set_prop_color(self, index, color):
        colors = self.model.shape_color.numpy()
        colors[self.prop_shapes[index]] = color
        self.model.shape_color.assign(colors)

    def update(self, poses):
        """Render every camera from link poses (xyzw); the scene's soil meshes must already be current.

        Fills ``rgb_images`` (H, W, 3 uint8) and ``depth_images`` (optical Z [m], NaN on a miss) by camera name.
        """
        for name, shape in self.soil_shapes.items():
            mesh = self.scene.meshes.get(name)
            self.pointers[shape] = mesh.id if mesh is not None else self.hidden
        self.model.shape_source_ptr.assign(self.pointers)
        self.state.body_q.assign(np.concatenate((np.asarray(poses, np.float32), self.prop_poses)))
        self.model.bvh_refit_shapes(self.state)
        for name, parent, xyz, quat, _, _ in self.cameras:
            pose = poses[self.names.index(parent)]
            mount = wp.transform(pose[:3], pose[3:]) * wp.transform(xyz, wp.quat(*quat) * OPTICAL_TO_GL)
            self.transform.assign([mount])
            self.sensor.update(
                self.state, self.transform, self.rays[name], color_image=self.color, forward_depth_image=self.depth
            )
            self.rgb_images[name] = self.color.numpy().view(np.uint8).reshape(self.height, self.width, 4)[..., :3]
            depth = self.depth.numpy()[0]
            depth[depth <= 0] = np.nan
            self.depth_images[name] = depth
        self.count += 1

    def draw(self, imgui, textures):
        if imgui.begin("RGB cameras (Newton Warp ray tracer)")[0]:
            for name, rgb in self.rgb_images.items():
                imgui.begin_group()
                imgui.text(f"{name}: {self.width}x{self.height}")
                textures.draw(imgui, f"rgb_{name}", rgb, self.count, (320, 240))
                imgui.end_group()
                imgui.same_line()
        imgui.end()
