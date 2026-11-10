"""Excavator gamepad teleop + cabin lidar height map + diggable sand, with Newton physics.

The Newton port of ../excavator_teleop.py + ../lidar_demo_script.py (which run PhysX + RTX lidar
in the Isaac Sim GUI). Newton's MuJoCo-Warp solver moves the excavator and rocks, the lidar is a
Newton BVH ray cast with the same mount, fan and range as the RTX one, and the sand is a light
height-grid model (sand.py) the bucket can dig, carry and dump.

Run from the Isaac Lab root, in either front end:

    uv run --extra isaacsim python scripts/sim-scratchpad/newton/excavator_lidar_teleop.py               # Kit (RTX)
    uv run --extra isaacsim python scripts/sim-scratchpad/newton/excavator_lidar_teleop.py --viz newton_gl

Options: ``--viz none`` runs headless (prints map stats), ``--lidar_step_deg 1`` sets ray spacing,
``--viewport_res 512`` the Kit render size, ``--no_rocks`` / ``--no_sand`` drop those.

ISO excavator layout (gamepad 0), same as ../excavator_teleop.py:
    left stick  X: swing (carriage)      left stick  Y: arm (tilt boom)
    right stick X: bucket (tool)         right stick Y: boom (lift)
    Y: clear the height map              B: reset pose, rocks and sand
Keyboard: A/D swing, W/S arm, I/K boom, J/L bucket, C clear, R reset.
"""

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heightmap as hm  # noqa: E402
from terrain import CACHE_PATH, build_terrain_patch  # noqa: E402

ROBOT_USD = os.path.join(HERE, "..", "..", "Isaac-hydraulic-actuator", "assets", "excavator_bucket.usd")

SIM_HZ = 120
RENDER_EVERY = 2  # at most 60 fps
MAX_STEPS_PER_FRAME = 8  # at least 15 fps when rendering can't keep up
MAX_LAG_S = 0.25
SAND_EVERY = 2  # sand rules at 60 Hz
LIDAR_HZ = 10.0
HEIGHTMAP_UPDATE_HZ = 5.0
LIDAR_FAR_M = 0.80
LIDAR_FOV_DEG = 60.0
# Mount on upper_carriage, copied from demo_env.usd's lidar_cabin: translate + rotateZYX(0, 33.8, 0).
LIDAR_POS = (0.1138096968191182, 0.08126573512700036, 0.17000010430812834)
LIDAR_PITCH_DEG = 33.8
POINT_RADIUS_M = 0.004
SAND_SHAPE_ID = -2  # lidar hits on the sand grid (not a Newton shape)
SLAB_AFTER_S = 0.3  # the 2D granular slab starts once the swing has been still this long
# Scripted cycle for --auto_dig: (start s, end s, joint, stick command).
AUTO_DIG = [
    (1.0, 1.9, "revolute_lift", 1.0),
    (1.9, 4.0, "revolute_tilt", -1.0),
    (1.9, 4.0, "revolute_tool", -1.0),
    (4.0, 6.0, "revolute_lift", -1.0),
    (6.0, 8.0, "revolute_carriage", 1.0),
    (8.5, 12.0, "revolute_tool", 1.0),
]

# Free base: spawned just above the highest bump under the tracks, then it settles onto the terrain.
# (demo_env.usd welds it at z=0.08, ~6 cm above.) The map's zero is the terrain's nominal level.
ROBOT_Z = 0.025
hm.GROUND_Z_M = -0.08
INIT_JOINT_POS_DEG = {"revolute_carriage": 0.0, "revolute_lift": -31.5, "revolute_tilt": 71.9, "revolute_tool": -43.2}
# Drive gains from excavator_bucket.usd (USD stores them per degree: 41.89 N*m/deg = 2400 N*m/rad).
ARM_GAINS = (2400.0, 120.0)
SWING_GAINS = (600.0, 40.0)

# joint: (stick axis, key +, key -, speed deg/s); signs match ../excavator_teleop.py.
JOINT_AXES = {
    "revolute_carriage": ("left_x", "D", "A", -15.0),
    "revolute_tilt": ("left_y", "W", "S", -12.5),
    "revolute_tool": ("right_x", "L", "J", -20.0),
    "revolute_lift": ("right_y", "I", "K", 10.0),
}
# Loose rocks: (x, y, edge m). They rest on the terrain, so the sand bed buries them until dug out.
ROCKS = [(0.42, -0.10, 0.05), (0.55, 0.08, 0.04), (0.62, -0.18, 0.06), (0.36, 0.17, 0.035)]
# Sand grid (1 cm cells) over the terrain, and the initial bed: a level fill with soft edges plus mounds.
SAND_GRID = dict(x0=-0.45, y0=-0.75, cell=0.01, nx=150, ny=150)
SAND_BED = dict(x_range=(0.15, 0.95), y_range=(-0.45, 0.45), level=-0.035, taper=0.06)
SAND_MOUNDS = [(0.62, 0.16, 0.05, 0.06), (0.45, -0.24, 0.03, 0.05)]
CAMERA_EYE = (1.35, -1.05, 0.85)
CAMERA_TARGET = (0.35, 0.0, 0.05)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--lidar_step_deg", type=float, default=1.0, help="Ray spacing; the RTX demo used 2.")
parser.add_argument("--no_rocks", action="store_true", help="Leave out the loose rocks.")
parser.add_argument("--no_sand", action="store_true", help="Leave out the sand bed.")
parser.add_argument(
    "--granular", action="store_true", help="Experimental: 2D MPM sand slab under the arm while digging."
)
parser.add_argument("--show_grains", action="store_true", help="Also draw the 2D slab's grains (debug).")
parser.add_argument("--auto_dig", action="store_true", help="Run a scripted dig, swing and dump cycle.")
parser.add_argument("--cameras", action="store_true", help="Kit: add the cabin and arm camera feeds.")
parser.add_argument("--camera_size", type=int, nargs=2, default=(512, 512), help="Feed width and height [px].")
parser.add_argument(
    "--feeds_only", action="store_true", help="With --cameras: stop rendering the perspective viewport (faster)."
)
parser.add_argument("--camera_hz", type=float, default=15.0, help="Feed rate [Hz].")
parser.add_argument("--record_dir", default=None, help="With --cameras: save frames + robot state here.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many physics steps.")
parser.add_argument(
    "--viewport_res", type=int, default=512, help="Kit: square viewport render resolution [px]; 0 keeps Kit's."
)
parser.add_argument("--no_realtime", action="store_true", help="Run as fast as possible instead of real time.")
add_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.visualizer is None:
    args_cli.visualizer = ["kit"]


def fan_pattern(cfg, device):
    """Solid-state lidar fan: a square azimuth x elevation grid around +X, like the RTX emitter table."""
    count = round(cfg.fov_deg / cfg.step_deg) + 1
    angles = torch.deg2rad(torch.linspace(-cfg.fov_deg / 2, cfg.fov_deg / 2, count, device=device))
    elevation, azimuth = torch.meshgrid(angles, angles, indexing="ij")
    directions = torch.stack(
        (torch.cos(elevation) * torch.cos(azimuth), torch.cos(elevation) * torch.sin(azimuth), torch.sin(elevation)),
        dim=-1,
    ).reshape(-1, 3)
    return torch.zeros_like(directions), directions


def create_cfgs():
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import RayCasterCfg
    from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
    from isaaclab.utils import configclass

    @configclass
    class FanPatternCfg(PatternBaseCfg):
        func = fan_pattern
        fov_deg: float = LIDAR_FOV_DEG
        step_deg: float = 2.0

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / SIM_HZ,
        render_interval=RENDER_EVERY,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            # Newton's collision pipeline handles the triangle-mesh terrain; MuJoCo's own would
            # convexify it. implicitfast keeps the stiff PD drives stable on the light links.
            solver_cfg=MJWarpSolverCfg(
                use_mujoco_contacts=False,
                integrator="implicitfast",
                njmax=400,
                nconmax=200,
                # Elliptic cones + high impratio stop the free base creeping down the bumps.
                cone="elliptic",
                impratio=10.0,
            ),
        ),
    )

    half = math.radians(LIDAR_PITCH_DEG) / 2
    scene_attrs = dict(
        light=AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=700.0, color=(0.85, 0.9, 1.0))
        ),
        sun=AssetBaseCfg(
            prim_path="/World/Sun",
            spawn=sim_utils.DistantLightCfg(intensity=1800.0, angle=1.0, color=(1.0, 0.95, 0.85)),
            init_state=AssetBaseCfg.InitialStateCfg(rot=(0.2706, 0.2706, 0.6533, 0.6533)),
        ),
        terrain=AssetBaseCfg(prim_path="/World/Terrain", spawn=sim_utils.UsdFileCfg(usd_path=CACHE_PATH)),
        robot=ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=os.path.abspath(ROBOT_USD)),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, ROBOT_Z),
                joint_pos={name: math.radians(deg) for name, deg in INIT_JOINT_POS_DEG.items()},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["revolute_lift", "revolute_tilt", "revolute_tool"],
                    stiffness=ARM_GAINS[0],
                    damping=ARM_GAINS[1],
                ),
                "swing": ImplicitActuatorCfg(
                    joint_names_expr=["revolute_carriage"], stiffness=SWING_GAINS[0], damping=SWING_GAINS[1]
                ),
            },
        ),
        lidar=RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/upper_carriage",
            offset=RayCasterCfg.OffsetCfg(pos=LIDAR_POS, rot=(0.0, math.sin(half), 0.0, math.cos(half))),
            ray_alignment="base",
            pattern_cfg=FanPatternCfg(step_deg=args_cli.lidar_step_deg),
            max_distance=LIDAR_FAR_M,
            update_period=1.0 / LIDAR_HZ,
            mesh_prim_paths=[],
        ),
    )
    if not args_cli.no_rocks:
        for index, (x, y, edge) in enumerate(ROCKS):
            scene_attrs[f"rock_{index}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Rock_{index}",
                spawn=sim_utils.CuboidCfg(
                    size=(edge, edge * 0.8, edge * 0.7),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    mass_props=sim_utils.MassCfg(density=2000.0),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.40, 0.38), roughness=0.9),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x, y, hm.GROUND_Z_M + 0.35 * edge + 0.02), rot=(0.0, 0.0, math.sin(index), math.cos(index))
                ),
            )
    if args_cli.cameras:
        from cameras import camera_cfgs

        scene_attrs.update(camera_cfgs(*args_cli.camera_size, args_cli.camera_hz))
    scene_cfg = configclass(type("ExcavatorSceneCfg", (InteractiveSceneCfg,), scene_attrs))(num_envs=1, env_spacing=0.0)
    return sim_cfg, scene_cfg


def to_numpy(value):
    value = getattr(value, "torch", value)
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value.numpy()


def make_sand(model, device):
    """Sand bed on the terrain: the base height per cell comes from casting rays straight down at it."""
    import newton
    import warp as wp
    from sand import SandBed

    g = SAND_GRID
    ys, xs = np.mgrid[0 : g["ny"], 0 : g["nx"]]
    origins = np.column_stack(
        [g["x0"] + (xs.ravel() + 0.5) * g["cell"], g["y0"] + (ys.ravel() + 0.5) * g["cell"], np.full(xs.size, 1.0)]
    ).astype(np.float32)
    count = len(origins)
    distance = wp.empty(count, dtype=wp.float32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=wp.array(origins, dtype=wp.vec3, device=device),
        ray_directions=wp.array(np.tile([0.0, 0.0, -1.0], (count, 1)).astype(np.float32), dtype=wp.vec3, device=device),
        ray_worlds=wp.full(count, -1, dtype=wp.int32, device=device),  # global world only: the terrain
        out_dist=distance,
    )
    distance = distance.numpy()
    base = np.where(distance >= 0.0, 1.0 - distance, hm.GROUND_Z_M).reshape(g["ny"], g["nx"])
    sand = SandBed(base, g["x0"], g["y0"], g["cell"], device)
    sand.fill_bed(**SAND_BED, mounds=SAND_MOUNDS)
    return sand


def run(sim_cfg, scene_cfg):
    from shape_id_lidar import ShapeIdRaycastSensor
    from teleop import Teleop

    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.utils import instantiate

    scene_cfg.lidar.class_type = ShapeIdRaycastSensor
    scene_cfg.terrain.spawn.usd_path = build_terrain_patch()  # needs pxr, so after launch
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = instantiate(scene_cfg)
    sim.reset()
    sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
    visualizers = {type(v).__name__: v for v in sim.visualizers}
    print(f"[Viz] {', '.join(visualizers) or 'none (headless)'}")
    if "KitVisualizer" in visualizers and args_cli.viewport_res > 0:
        from omni.kit.viewport.utility import get_active_viewport

        # RTX cost scales with pixels; a small viewport keeps the demo responsive on a laptop GPU.
        get_active_viewport().resolution = (args_cli.viewport_res, args_cli.viewport_res)
        if args_cli.cameras and args_cli.feeds_only:
            get_active_viewport().updates_enabled = False  # only the camera feeds are rendered

    robot, lidar = scene["robot"], scene["lidar"]
    rocks = [scene[name] for name in scene.keys() if name.startswith("rock_")]
    device = sim.device
    joint_ids = {name: robot.joint_names.index(name) for name in JOINT_AXES}
    bucket_id = robot.body_names.index("bucket")
    limits = to_numpy(robot.data.joint_pos_limits)[0]
    default_pos = torch.tensor(to_numpy(robot.data.default_joint_pos), device=device)
    targets = default_pos.clone()
    self_shapes = lidar.shapes_under("/World/envs/env_0/Robot/")
    print(f"[Newton] Joints {robot.joint_names}; fixed base: {robot.is_fixed_base}.")
    print(
        f"[Lidar] {lidar.num_rays} rays ({args_cli.lidar_step_deg:g} deg), 0-{LIDAR_FAR_M} m, {LIDAR_HZ:g} Hz; "
        f"{int(self_shapes.sum())} of {len(self_shapes)} shapes are the excavator."
    )

    sand = bucket = slab = None
    sand_mesh = load_mesh = grains = None
    if not args_cli.no_sand:
        from sand import Bucket
        from sand_render import MeshPublisher

        sand, bucket = make_sand(lidar.backend.model, device), Bucket()
        print(
            f"[Sand] {sand.nx}x{sand.ny} cells of {sand.cell * 100:g} cm, {sand.volume() * 1000:.2f} L in the bed; "
            f"bucket holds {bucket.capacity * 1000:.2f} L."
        )
        if args_cli.granular:
            from sand2d import GRAIN_RADIUS, ArmSlab

            slab = ArmSlab(sand, bucket, device)
        if visualizers:
            from sand_render import PointsPublisher

            sand_mesh = MeshPublisher(sim.visualizers, "/World/Sand")
            load_mesh = MeshPublisher(sim.visualizers, "/World/BucketLoad")
            if slab is not None and args_cli.show_grains:
                grains = PointsPublisher(sim.visualizers, "/World/SandGrains", GRAIN_RADIUS)

    markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/LidarPoints",
            markers={
                "terrain": sim_utils.SphereCfg(
                    radius=POINT_RADIUS_M,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.3), emissive_color=(0.0, 4.0, 1.2)
                    ),
                ),
                "self": sim_utils.SphereCfg(
                    radius=POINT_RADIUS_M,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.35, 0.0), emissive_color=(4.0, 1.4, 0.0)
                    ),
                ),
            },
        )
    )
    if args_cli.cameras:
        markers.set_visibility(False)  # the debug points would show up in the camera images
    view = None
    if "KitVisualizer" in visualizers:
        from heightmap_view import HeightMapView

        view = HeightMapView()
    elif "NewtonGLVisualizer" in visualizers:
        from heightmap_panel import HeightMapPanel

        view = HeightMapPanel(visualizers["NewtonGLVisualizer"])
    teleop = Teleop(JOINT_AXES, sim.visualizers)
    feeds = {name[: -len("_cam")]: scene[name] for name in scene.keys() if name.endswith("_cam")}
    feed_window = recorder = None
    if feeds:
        from cameras import CameraFeedWindow, FrameRecorder, rgb_frame

        if "KitVisualizer" in visualizers:
            feed_window = CameraFeedWindow(feeds, *args_cli.camera_size)
        if args_cli.record_dir:
            recorder = FrameRecorder(args_cli.record_dir)
        print(f"[Cameras] {', '.join(feeds)} at {args_cli.camera_size[0]}x{args_cli.camera_size[1]}, {args_cli.camera_hz:g} Hz")
    last_feed_t = -1.0
    memory = hm.new_heightmap_memory()

    def reset():
        nonlocal memory
        targets.copy_(default_pos)
        robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device))
        robot.write_joint_position_to_sim_index(position=default_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default_pos))
        for rock in rocks:
            rock.reset()
            rock.write_root_pose_to_sim_index(root_pose=rock.data.default_root_pose.torch.clone())
            rock.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device))
        if slab is not None:
            slab.discard()
        if sand is not None:
            sand.fill_bed(**SAND_BED, mounds=SAND_MOUNDS)
            bucket.load = 0.0
        memory = hm.new_heightmap_memory()

    def body_pose(body_id):
        pos = to_numpy(robot.data.body_link_pos_w)[0, body_id].astype(np.float64)
        quat = to_numpy(robot.data.body_link_quat_w)[0, body_id].astype(np.float64)
        return pos, quat

    def bucket_pose():
        return body_pose(bucket_id)

    carriage_id = robot.body_names.index("upper_carriage")
    swing_still_s = 0.0

    dt = sim.get_physics_dt()
    step, last_render_step, last_map_t, last_marker_t = 0, 0, -1.0, -1.0
    sand_version_drawn = -1
    wall_start = time.monotonic()
    while sim.is_running() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
        # Rate control: each joint target moves at stick * speed and holds when released.
        commands = teleop.axes()
        if args_cli.auto_dig:
            t = (step * dt) % 14.0
            for t0, t1, name, command in AUTO_DIG:
                if t0 <= t < t1:
                    commands[name] = command
        for name, command in commands.items():
            if command:
                j = joint_ids[name]
                speed = math.radians(JOINT_AXES[name][3])
                targets[0, j] = float(np.clip(float(targets[0, j]) + command * speed * dt, *limits[j]))
        if teleop.pressed("B", "R"):
            reset()
            print("[Teleop] Reset pose, rocks and sand.")
        if teleop.pressed("Y", "C"):
            memory = hm.new_heightmap_memory()
            print("[Teleop] Height map cleared.")
        robot.actuators.target_command.set_position_index(value=targets, full_data=True)

        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        step += 1
        sim_t = step * dt

        if slab is not None:
            # The 2D granular slab runs while the swing is still; swinging hands back to the height grid.
            swing_still_s = 0.0 if commands.get("revolute_carriage") else swing_still_s + dt
            if slab.active and swing_still_s == 0.0:
                slab.deactivate()
            elif not slab.active and swing_still_s >= SLAB_AFTER_S:
                slab.activate(*body_pose(carriage_id), *bucket_pose())
        if sand is not None and step % SAND_EVERY == 0:
            if slab is not None and slab.active:
                slab.step(*bucket_pose(), SAND_EVERY * dt)
            else:
                bucket.update(sand, *bucket_pose(), SAND_EVERY * dt)
                sand.relax(2)

        if sim_t - last_marker_t >= 1.0 / LIDAR_HZ or sim_t - last_map_t >= 1.0 / HEIGHTMAP_UPDATE_HZ:
            starts, directions, distance, shape_ids = lidar.scan()
            if sand is not None:
                sand_t = sand.raycast(starts, directions, np.minimum(distance, LIDAR_FAR_M))
                on_sand = sand_t >= 0.0
                distance = np.where(on_sand, sand_t, distance)
                shape_ids = np.where(on_sand, SAND_SHAPE_ID, shape_ids)
            hit = np.isfinite(distance)
            hits = np.where(hit[:, None], starts + np.where(hit, distance, 0.0)[:, None] * directions, np.inf)
            on_self = hit & (shape_ids >= 0) & self_shapes[np.maximum(shape_ids, 0)]
            ground = hit & ~on_self
            if sim_t - last_marker_t >= 1.0 / LIDAR_HZ:
                last_marker_t = sim_t
                shown = hit if hit.any() else np.ones_like(hit)  # keep at least one instance alive
                markers.visualize(
                    translations=np.nan_to_num(hits[shown], posinf=-10.0),
                    marker_indices=on_self[shown].astype(np.int64),
                )
            if sim_t - last_map_t >= 1.0 / HEIGHTMAP_UPDATE_HZ:
                last_map_t = sim_t
                memory, cleared = hm.update_heightmap(memory, starts[0], hits[ground], int(sim_t * 1e9))
                status = (
                    f"Live | t={sim_t:6.1f} s | {int(ground.sum())} terrain returns, "
                    f"{int(on_self.sum())} on excavator, {cleared} stale cells cleared"
                )
                if sand is not None:
                    status += f" | sand {sand.volume() * 1000:.2f} L + bucket {bucket.load * 1000:.2f} L"
                    if slab is not None and slab.active:
                        status += f" | 2D slab: {slab.count} grains, {slab.bucket_volume() * 1000:.2f} L in bucket"
                if view is not None:
                    view.update(memory)
                    view.set_status(status)
                elif step % int(SIM_HZ) < int(SIM_HZ / HEIGHTMAP_UPDATE_HZ):
                    coverage = 100 * memory["valid"].mean()
                    measured = memory["height_m"][memory["valid"]]
                    span = f"{measured.min() * 1000:+.0f}..{measured.max() * 1000:+.0f} mm" if measured.size else "-"
                    joints = " ".join(f"{math.degrees(v):+6.1f}" for v in to_numpy(robot.data.joint_pos)[0])
                    print(f"[Map] {status} | coverage {coverage:.1f}% height {span} | joints(deg) {joints}")

        if feeds and sim_t - last_feed_t >= 1.0 / args_cli.camera_hz:
            last_feed_t = sim_t
            frames = {name: rgb_frame(camera) for name, camera in feeds.items()}
            if feed_window is not None:
                feed_window.update(frames)
            if recorder is not None:
                recorder.write(
                    frames,
                    {
                        "t": round(sim_t, 4),
                        "joint_names": list(JOINT_AXES),
                        "joint_pos": [float(to_numpy(robot.data.joint_pos)[0, joint_ids[n]]) for n in JOINT_AXES],
                        "joint_target": [float(targets[0, joint_ids[n]]) for n in JOINT_AXES],
                        "command": [float(commands.get(n, 0.0)) for n in JOINT_AXES],
                    },
                )

        # Like the Isaac Sim timeline: physics keeps up with the clock and frames are drawn when time allows.
        # A render (~25 ms at 512 px in Kit on a laptop GPU) costs 100x a physics step, so rendering on a
        # fixed step count would slow the sim itself down to the frame rate.
        since_render = step - last_render_step
        if args_cli.no_realtime:
            render_now = since_render >= RENDER_EVERY
        else:
            lag = time.monotonic() - wall_start - sim_t  # > 0: the sim is behind the clock
            if lag > MAX_LAG_S:  # after a stall (window drag, map rebuild), don't fast-forward
                wall_start += lag - MAX_LAG_S
            render_now = (since_render >= RENDER_EVERY and lag <= 0.0) or since_render >= MAX_STEPS_PER_FRAME
            if lag < 0.0:
                time.sleep(-lag)
        if sim.is_rendering and render_now:
            if sand_mesh is not None:
                if sand.version != sand_version_drawn:
                    sand_mesh.update(*sand.mesh())
                    sand_version_drawn = sand.version
                if grains is not None:
                    grains.update(*slab.grains_world())
                in_slab = slab.bucket_volume() if slab is not None and slab.active else None
                carried = bucket.load_mesh(*bucket_pose(), load=in_slab)
                load_mesh.update(*(carried if carried is not None else (None, None)))
            sim.render()
            last_render_step = step
    if hasattr(view, "destroy"):
        view.destroy()
    if feed_window is not None:
        feed_window.destroy()
    if recorder is not None:
        recorder.close()
    return memory


def main():
    sim_cfg, scene_cfg = create_cfgs()
    with launch_simulation(SimpleNamespace(sim=sim_cfg, scene=scene_cfg), args_cli):
        run(sim_cfg, scene_cfg)


if __name__ == "__main__":
    main()
