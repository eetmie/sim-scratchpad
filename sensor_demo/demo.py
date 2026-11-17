# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed excavator, hybrid soil, geometric lidars and cabin heightmaps. See README.md."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "newton"))
from diagnostics import Recorder, capture_viewport
from soil import COLOR, Soil, SoilConfig
from task import (
    BUCKET_UP_M,
    COLORS,
    DIG_HOLD_S,
    DIG_SUCCESS_L,
    DIG_TIMEOUT_S,
    STILL_DEG_S,
    SUCCESS_L,
    containers,
    lifted,
    new_episode,
    parked,
)
from teleop import Teleop

GROUND_Z = SoilConfig.ground  # the physics ground and the soil's hard ground must agree
ROBOT_USD = HERE.parent.parent / "Isaac-hydraulic-actuator/assets/excavator_bucket.usd"
JOINT_AXES = {
    "revolute_carriage": ("left_x", "D", "A", -15.0),
    "revolute_tilt": ("left_y", "W", "S", -12.5),
    "revolute_tool": ("right_x", "L", "J", -20.0),
    "revolute_lift": ("right_y", "I", "K", 10.0),
}
INITIAL_DEG = {
    "revolute_carriage": 0.0,
    "revolute_lift": -31.5,
    "revolute_tilt": 71.9,
    "revolute_tool": -43.2,
}
# Stroke keyframes (s, joint degrees): open and extend, teeth-first penetration,
# drag toward the machine while curling, then lift and swing with the arm out so the dump clears the base.
AUTO_DIG = [
    (0.5, dict(INITIAL_DEG)),
    (1.7, dict(INITIAL_DEG, revolute_tilt=60.0, revolute_tool=-67.0)),
    (3.5, dict(INITIAL_DEG, revolute_tilt=60.0, revolute_tool=-67.0, revolute_lift=-14.0)),
    (6.0, dict(INITIAL_DEG, revolute_tilt=88.0, revolute_tool=-37.0, revolute_lift=-16.0)),
    (7.6, dict(INITIAL_DEG, revolute_tilt=90.0, revolute_tool=-5.0, revolute_lift=-26.0)),
    (8.4, dict(INITIAL_DEG, revolute_tilt=90.0, revolute_tool=-5.0, revolute_lift=-34.0)),
    (10.4, dict(INITIAL_DEG, revolute_tilt=55.0, revolute_tool=20.0, revolute_lift=-38.0, revolute_carriage=-45.0)),
    (13.6, dict(INITIAL_DEG, revolute_tilt=55.0, revolute_tool=-45.0, revolute_lift=-38.0, revolute_carriage=-45.0)),
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after N 120 Hz physics steps.")
    parser.add_argument(
        "--auto_cycles",
        type=int,
        default=1,
        help="With --auto_dig, repeat on the evolving bed; cycles include 8 s to return the arm.",
    )
    parser.add_argument("--render_hz", type=float, default=60, help="Maximum viewport frame rate.")
    parser.add_argument(
        "--auto_dig",
        action="store_true",
        help="Run one scripted 14-second dig/swing/dump cycle, then hold.",
    )
    parser.add_argument("--no_realtime", action="store_true", help="Run without wall-clock throttling.")
    parser.add_argument("--viewport_res", type=int, default=720, help="Kit viewport size in pixels.")
    parser.add_argument(
        "--particles",
        type=int,
        default=10000,
        help="Maximum simultaneous dynamic soil particles.",
    )
    parser.add_argument(
        "--cell",
        type=float,
        default=0.01,
        help="Resting grid and nominal particle spacing [m].",
    )
    parser.add_argument("--substeps", type=int, default=4, help="Particle substeps per 60 Hz soil tick.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="Contact iterations per particle substep.",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.7,
        help="Particle and bucket friction coefficient.",
    )
    parser.add_argument(
        "--cohesion",
        type=float,
        default=0.0,
        help="Weak particle attraction, 0 (dry) to 1.",
    )
    parser.add_argument("--bucket_margin", type=float, default=0.003, help="Bucket shell collision margin [m].")
    parser.add_argument(
        "--soil_render",
        choices=("surface", "grains", "both"),
        default="surface",
        help="Smooth opaque sand surface, individual grains, or both.",
    )
    parser.add_argument(
        "--surface_voxel",
        type=float,
        default=None,
        help="Surface reconstruction voxel size [m]; defaults to half --cell.",
    )
    parser.add_argument(
        "--surface_blend",
        type=float,
        default=None,
        help="Width of the render-only ground/particle transition [m]; default 1.5 cells.",
    )
    parser.add_argument("--screenshot", type=Path, help="Save the final Newton viewer frame as a PNG.")
    parser.add_argument("--record_dir", type=Path, help="Write per-tick CSV, jump snapshots and viewport PNGs here.")
    parser.add_argument("--capture_every", type=float, default=1.0, help="Seconds between recorded frames; 0 disables.")
    parser.add_argument("--debug_particles", action="store_true", help="Highlight fast upward/sideways grains in red.")
    parser.add_argument(
        "--view",
        choices=("overview", "soil", "cabin"),
        default="overview",
        help="Initial viewport framing, or 'cabin': show the cabin RGB camera instead of drawing the 3D scene.",
    )
    parser.add_argument("--lidar_hz", type=float, default=5.0, help="Height map (and lidar) rate; simulation Hz.")
    parser.add_argument("--lidar_step_deg", type=float, default=1.0, help="60 degree square fan spacing.")
    parser.add_argument("--lidar_range", type=float, default=0.8, help="Maximum geometric lidar range [m].")
    parser.add_argument(
        "--show_lidar", action="store_true", help="Draw soil (green), moving sand (yellow), self hits (red)."
    )
    parser.add_argument("--depth_cameras", action="store_true", help="Add lightweight 160x120 geometric depth cameras.")
    parser.add_argument(
        "--rgb_cameras",
        nargs="*",
        choices=("cabin", "arm"),
        help="Add RGB-D cameras (Newton ray tracer): name them, or give no name for both.",
    )
    parser.add_argument("--lidars", action="store_true", help="Add the two geometric lidars and their height map.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Randomise the sand surface and the container (which one, its colour and pose), anew on every reset.",
    )
    parser.add_argument("--containers", type=int, help="Spawn only the first N containers of task.ASSETS.")
    parser.add_argument(
        "--expert", action="store_true", help="A scripted operator digs and fills the container, episode after episode."
    )
    parser.add_argument(
        "--task",
        choices=("move", "dig"),
        default="move",
        help="move: dig and pour into the container. dig: heaps on a bare pit, one cut lifted clear, then a reset.",
    )
    parser.add_argument(
        "--dig_probe",
        type=float,
        metavar="DEG",
        help="With --task dig and --seed: a test layout, one heap at this angle (left positive) on a bare pit.",
    )
    parser.add_argument("--episodes", type=int, help="With --expert: stop after N episodes.")
    parser.add_argument("--scoops", type=int, default=3, help="With --expert: dig-and-pour loops per episode.")
    parser.add_argument(
        "--ros", action="store_true", help="Play the robot on the ROS 2 contract topics (start with run_ros.sh)."
    )
    parser.add_argument(
        "--showcase",
        action="store_true",
        help="Stand every container beside the pit at once, each in its own colour, to look at them.",
    )
    parser.add_argument("--camera_hz", type=float, default=30.0, help="RGB camera rate in simulation time.")
    parser.add_argument("--rgb_width", type=int, default=640, help="RGB image width [px]; 4:3 like the real sensor.")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.visualizer is None:
        args.visualizer = ["newton_gl"]
    args.lidars |= args.show_lidar
    if args.rgb_cameras == []:
        args.rgb_cameras = ["cabin", "arm"]
    if (args.view == "cabin" or args.ros) and "cabin" not in (args.rgb_cameras or ()):
        args.rgb_cameras = ["cabin", *(args.rgb_cameras or ())]
    if not 0 < args.lidar_hz <= 60 or not 0.25 <= args.lidar_step_deg <= 10 or args.lidar_range <= 0.01:
        parser.error("lidar_hz must be in (0,60], step_deg in [0.25,10], range > 0.01")
    if not 0 < args.camera_hz <= 120 or args.rgb_width < 16:
        parser.error("camera_hz must be in (0,120] and rgb_width at least 16")
    if args.particles < 1 or args.cell <= 0 or args.substeps < 1 or args.iterations < 1:
        parser.error("particles, cell, substeps and iterations must be positive")
    if args.auto_cycles < 1 or not 0 < args.render_hz <= 120:
        parser.error("auto_cycles must be positive and render_hz must be in (0, 120]")
    if args.friction < 0 or not 0 <= args.cohesion <= 1:
        parser.error("friction must be nonnegative and cohesion must be between 0 and 1")
    if args.bucket_margin < 0 or (args.surface_voxel is not None and args.surface_voxel <= 0):
        parser.error("bucket_margin must be nonnegative and surface_voxel must be positive")
    if args.surface_blend is not None and args.surface_blend <= 0:
        parser.error("surface_blend must be positive")
    if not ROBOT_USD.is_file():
        parser.error(f"Excavator asset not found: {ROBOT_USD}")
    args.assets = containers(args.containers)
    if not args.assets:
        parser.error("No container asset found under assets/simready (see the download command in task.py)")
    return args


def spawn_fixed_excavator(prim_path, cfg, translation=None, orientation=None):
    """Anchor the asset's root body before the scene's Newton import runs."""
    from isaaclab.sim import get_current_stage
    from isaaclab.sim.schemas.schemas import create_world_fixed_joint
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    from pxr import UsdGeom, UsdPhysics

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    stage = get_current_stage()
    create_world_fixed_joint(stage.GetPrimAtPath(f"{prim.GetPath()}/lower_carriage"), stage)
    # Invisible hard ground at the brim of the pit, to carry the container. The machine must not hit it (the
    # bucket digs below ground level), and Newton only resolves a collision filter inside one imported asset,
    # so the plane lives under the robot prim.
    links = [link for link in prim.GetChildren() if link.HasAPI(UsdPhysics.RigidBodyAPI)]
    # A plane, not a box: a container the bucket presses into the ground must come back up, not out underneath.
    slab = UsdGeom.Plane.Define(stage, f"{prim.GetPath()}/ground_plane")
    slab.CreateAxisAttr("Z")
    slab.AddTranslateOp().Set((0.0, 0.0, GROUND_Z - translation[2]))
    slab.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(slab.GetPrim())
    for link in links:
        UsdPhysics.FilteredPairsAPI.Apply(link).CreateFilteredPairsRel().AddTarget(slab.GetPath())
    return prim


def configs(args):
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    soil_cfg = SoilConfig(
        cell=args.cell,
        capacity=args.particles,
        substeps=args.substeps,
        iterations=args.iterations,
        friction=args.friction,
        cohesion=args.cohesion,
        bucket_margin=args.bucket_margin,
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        device=args.device,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                integrator="implicitfast",
                use_mujoco_contacts=False,
                njmax=3000 if args.showcase else 600,  # every container on the ground at once needs more
                nconmax=1000 if args.showcase else 200,
            )
        ),
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=900))
        sun = AssetBaseCfg(
            prim_path="/World/Sun",
            spawn=sim_utils.DistantLightCfg(intensity=1800, angle=2, color=(1.0, 0.94, 0.83)),
            init_state=AssetBaseCfg.InitialStateCfg(rot=(0.2706, 0.2706, 0.6533, 0.6533)),
        )
        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.CuboidCfg(
                size=(8, 8, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.19, 0.20, 0.21), roughness=0.95),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, soil_cfg.floor - 0.026)),
        )
        pedestal = AssetBaseCfg(
            prim_path="/World/Pedestal",
            spawn=sim_utils.CuboidCfg(
                size=(0.36, 0.30, -soil_cfg.floor),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.23, 0.24), roughness=0.9),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, soil_cfg.floor / 2)),
        )
        # Every container is in the scene from the start; each episode stands one by the pit and parks the rest.
        containers = RigidObjectCollectionCfg(
            rigid_objects={
                f"container_{i}": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Container_{i}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=str(asset.path), variants={"Newton": "Enabled", "PhysX": "Disabled"}
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(parked(i, soil_cfg.ground)[:3].tolist())),
                )
                for i, asset in enumerate(args.assets)
            }
        )
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ROBOT_USD),
                func=spawn_fixed_excavator,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0, 0, 0.08),
                joint_pos={k: math.radians(v) for k, v in INITIAL_DEG.items()},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "revolute_lift",
                        "revolute_tilt",
                        "revolute_tool",
                    ],
                    stiffness=2400,
                    damping=120,
                ),
                "swing": ImplicitActuatorCfg(joint_names_expr=["revolute_carriage"], stiffness=600, damping=40),
            },
        )

    return sim_cfg, SceneCfg(num_envs=1, env_spacing=0), soil_cfg


def numpy(value):
    value = getattr(value, "torch", value)
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value.numpy()


def scripted_commands(sim_time, cycles, targets, default, joint_ids):
    """Repeat without resetting soil; return the tool gently between strokes."""
    commands = dict.fromkeys(JOINT_AXES, 0.0)
    if cycles == 1:
        phase = sim_time
    else:
        if int(sim_time / 22) >= cycles:
            return commands
        phase = sim_time % 22
    if phase < 14:
        times = [t for t, _ in AUTO_DIG]
        goals = {name: math.radians(np.interp(phase, times, [k[name] for _, k in AUTO_DIG])) for name in joint_ids}
        gain = 0.1
    elif cycles > 1:
        goals = {name: float(default[0, j]) for name, j in joint_ids.items()}
        gain = 0.75
    else:
        return commands
    for name, j in joint_ids.items():
        speed = math.radians(JOINT_AXES[name][3])
        commands[name] = float(np.clip((goals[name] - float(targets[0, j])) / (gain * speed), -1, 1))
    return commands


def create_sensors(args, soil, robot, sim, drawn):
    """``drawn``: the visualizers that draw the 3D scene (none when the viewport shows the cabin camera)."""
    from sand_render import PointsPublisher

    grains = None
    if args.soil_render in ("grains", "both") or args.debug_particles:
        grains = PointsPublisher(drawn, "/World/SoilGrains", radius=soil.cfg.cell * 0.535)
    from lidar import SensorRig
    from surface import SandSurface

    # Sensors need the same surface even when no viewport is requested.
    surface = SandSurface(
        drawn if args.soil_render in ("surface", "both") else [], soil, args.surface_voxel, args.surface_blend
    )
    rig = SensorRig(soil, robot.body_names, sim.visualizers, args)
    lidar_points = PointsPublisher(drawn, "/Visuals/Lidar", radius=0.002) if args.show_lidar else None
    cameras = None
    if args.rgb_cameras:
        from rgb import CameraRig, robot_visuals

        parts = robot_visuals(ROBOT_USD, robot.body_names)
        cameras = CameraRig(
            rig.scene,
            parts,
            robot.body_names,
            soil.cfg.floor,
            args.rgb_width,
            cameras=args.rgb_cameras,
            # The containers, then the pad under each one.
            props=[(c.points, c.faces, COLORS["white"]) for c in args.assets]
            + [(c.pad, c.pad_faces, tuple(COLOR)) for c in args.assets],
        )
        for visualizer in sim.visualizers:
            if type(visualizer).__name__ == "NewtonGLVisualizer":
                visualizer.register_ui_callback(lambda imgui: cameras.draw(imgui, rig.textures), position="free")
    frames = None
    if cameras is not None and args.record_dir:
        from cameras import FrameRecorder  # ../newton: PNG per camera plus steps.jsonl

        frames = FrameRecorder(str(args.record_dir / "cameras"))
    return grains, surface, rig, lidar_points, cameras, frames


def update_sensors(scan_now, camera_now, rig, cameras, frames, lidar_points, surface, robot, sim_time):
    """Scan the lidars and/or render the cameras from one robot pose; return the camera cost [ms], if any."""
    poses = np.concatenate((numpy(robot.data.body_link_pos_w)[0], numpy(robot.data.body_link_quat_w)[0]), axis=1)
    if scan_now:
        rig.update(surface, poses, sim_time)
        if lidar_points is not None:
            lidar_points.update(rig.points, rig.colors)
    else:
        rig.sync_soil(surface)
    if not camera_now:
        return []
    begin = time.perf_counter()
    cameras.update(poses)
    if frames:
        joints = numpy(robot.data.joint_pos)[0]
        record = {"time_s": sim_time, "joint_pos": dict(zip(robot.joint_names, joints.tolist()))}
        frames.write(cameras.rgb_images, record)
    return [(time.perf_counter() - begin) * 1000]


def run(args, sim_cfg, scene_cfg, soil_cfg):
    import warp as wp
    from sand_render import MeshPublisher

    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    sim = sim_utils.SimulationContext(sim_cfg)
    scene = instantiate(scene_cfg)
    sim.reset()
    eye, target = ((1.15, -0.85, 0.65), (0.40, 0, 0.02))
    if args.view == "soil":
        eye, target = (0.95, -0.48, 0.36), (0.55, 0, -0.01)
    sim.set_camera_view(eye=eye, target=target)
    robot = scene["robot"]
    if not robot.is_fixed_base:
        raise RuntimeError("The soil demo requires the excavator root to be fixed to the world.")
    visualizers = {type(v).__name__: v for v in sim.visualizers}
    if "KitVisualizer" in visualizers and args.viewport_res > 0:
        from omni.kit.viewport.utility import get_active_viewport

        get_active_viewport().resolution = (args.viewport_res, args.viewport_res)
    soil = Soil(soil_cfg, device=sim.device)
    recorder = Recorder(args.record_dir, soil, args.capture_every) if args.record_dir else None
    viewer = getattr(visualizers.get("NewtonGLVisualizer"), "_viewer", None)
    # The cabin view replaces the 3D scene with the camera image, so nothing needs to be staged for drawing.
    drawn = [] if args.view == "cabin" else sim.visualizers
    mesh = MeshPublisher(drawn, "/World/Soil", color=tuple(float(c) for c in COLOR))
    assets, count = args.assets, len(args.assets)
    bodies = scene["containers"]  # the viewer draws them from the physics model
    model = sim.backend_manager.backend.model
    shapes = [[s for s, label in enumerate(model.shape_label) if f"/Container_{i}/" in label] for i in range(count)]
    container_pad = MeshPublisher(drawn, "/World/ContainerPad", color=tuple(float(c) for c in COLOR))
    grains, surface, rig, lidar_points, cameras, frames = create_sensors(args, soil, robot, sim, drawn)
    container = assets[0]
    if args.view == "cabin" and viewer is not None:
        end_frame = viewer.end_frame

        def end_frame_with_camera():
            if "cabin" in cameras.rgb_images:
                viewer.log_image("cabin", np.ascontiguousarray(cameras.rgb_images["cabin"]), fullscreen=True)
            end_frame()

        viewer.end_frame = end_frame_with_camera
    rng = None if args.seed is None else np.random.default_rng(args.seed)
    bridge = None
    if args.ros:
        from ros_bridge import RosBridge

        bridge = RosBridge(JOINT_AXES, robot.joint_names, args.rgb_cameras)

    def result():
        """How the running episode went: the resting sand in the container."""
        strokes = {"strokes": expert.strokes} if expert else {}  # (azimuth deg, start m, surface far, near m)
        if args.task == "dig":
            litres, up = lifted(soil) * 1000, float(bucket_pose()[2] - soil.cfg.ground)
            success = litres >= DIG_SUCCESS_L and up >= BUCKET_UP_M
            return {"lifted_l": round(litres, 4), "bucket_up_m": round(up, 3), "success": success, **strokes}
        litres = container.volume(soil) * 1000
        return {"in_container_l": round(litres, 4), "success": litres >= SUCCESS_L, **strokes}

    def show_container():
        active = assets.index(container)
        pad = container.pad_pose(soil.cfg.ground)
        container_pad.update(container.pad + pad[:3], container.pad_faces)
        if cameras is not None:
            cameras.prop_poses[:] = [parked(i, soil.cfg.ground) for i in range(2 * count)]
            cameras.prop_poses[active], cameras.prop_poses[count + active] = container.pose, pad

    def place(ids, poses):
        bodies.write_body_link_pose_to_sim_index(
            body_poses=torch.tensor(np.asarray(poses, np.float32)[None], device=sim.device), body_ids=ids
        )
        bodies.write_body_link_velocity_to_sim_index(
            body_velocities=torch.zeros((1, len(ids), 6), device=sim.device), body_ids=ids
        )

    def park():
        """Put the unused containers back at their place in the air (see task.parked)."""
        ids = [i for i in range(count) if assets[i] is not container]
        if ids and not args.showcase:
            place(ids, [parked(i, soil.cfg.ground) for i in ids])

    def episode():
        """Refill the pit and stand one container up beside it, the rest parked; random when --seed is given."""
        nonlocal container, started, held
        if bridge:
            bridge.end_episode(step / 120, result())
        started, held = step, 0
        active, layout = new_episode(soil, assets, rng, dig=args.task == "dig", probe=args.dig_probe)
        container = assets[active]
        place([active], [container.pose])
        park()
        colours = {active: layout["colour"]}
        if args.showcase:
            # Rows of five on both sides of the pit, 30 cm apart. The sand follows the first one only.
            colours = {i: list(COLORS)[i % len(COLORS)] for i in range(count)}
            rows = [(-0.35 + 0.3 * (i % 5), (0.55 + 0.3 * (i // 10)) * (1 if i // 5 % 2 == 0 else -1)) for i in range(count)]
            place(list(range(count)), [(x, y, soil.cfg.ground, 0, 0, 0, 1) for x, y in rows])
            for i, (x, y) in enumerate(rows):
                print(f"[Showcase] x={x:+.2f} y={y:+.2f}  {colours[i]:7s} {assets[i].noun:11s} {assets[i].name}")
        colors = model.shape_color.numpy()
        for i, colour in colours.items():
            colors[shapes[i]] = COLORS[colour]
            if cameras is not None:
                cameras.set_prop_color(i, COLORS[colour])
        model.shape_color.assign(colors)
        show_container()
        rig.observed.fill(np.nan)
        rig.oracle.fill(np.nan)
        if args.record_dir:
            with (args.record_dir / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **layout}) + "\n")
        if bridge:
            bridge.start_episode(step / 120, layout)
        if expert:
            joints = [float(targets[0, joint_ids[name]]) for name in JOINTS]
            if args.task == "dig":
                expert.start(soil, container, joints, scoops=1, pour=False)
            else:
                expert.start(soil, container, joints, args.scoops)
        print(f"[Episode] {layout}")

    teleop = Teleop(JOINT_AXES, sim.visualizers)
    if args.screenshot and "NewtonGLVisualizer" not in visualizers:
        raise ValueError("--screenshot requires --viz newton_gl.")
    joint_ids = {name: robot.joint_names.index(name) for name in JOINT_AXES}
    arm_ids = list(joint_ids.values())
    bucket_id = robot.body_names.index("bucket")
    limits = numpy(robot.data.joint_pos_limits)[0]
    if bridge:
        limits = bridge.limits(limits)
    expert, finished = None, 0
    if args.expert:
        from expert import JOINTS, Expert

        seed = None if args.seed is None else args.seed + 1  # its own stream: the layouts stay those of --seed
        expert = Expert(JOINT_AXES, [limits[joint_ids[name]] for name in JOINTS], np.random.default_rng(seed))
    default = torch.tensor(numpy(robot.data.default_joint_pos), device=sim.device)
    targets = default.clone()

    def bucket_pose():
        return np.concatenate(
            (
                numpy(robot.data.body_link_pos_w)[0, bucket_id],
                numpy(robot.data.body_link_quat_w)[0, bucket_id],
            )
        )

    def reset():
        targets.copy_(default)
        robot.write_joint_position_to_sim_index(position=default)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default))
        robot.reset()
        scene.update(0)
        episode()

    print(
        f"[Soil] Fixed excavator | {sim.device} | {soil.n}x{soil.n} cells | "
        f"{soil.initial_volume * 1000:.2f} L | up to {soil_cfg.capacity} moving particles"
    )
    print("[Controls] Left: swing/arm. Right: bucket/boom. B or R: reset. Y or C: reset soil only.")
    print("[Keyboard] A/D swing, W/S arm, I/K boom, J/L bucket. Mouse controls the view.")
    print(f"[Soil] Initial bucket xyz: {bucket_pose()[:3].round(3)}")
    step = started = held = 0
    episode()
    mesh.update(*soil.mesh())
    drawn_version, last_render = soil.version, 0
    wall_start = run_start = time.monotonic()
    soil_times = []
    render_every = max(1, round(120 / args.render_hz))
    scan_every = max(1, round(120 / args.lidar_hz))
    camera_every = max(1, round(120 / args.camera_hz))
    camera_times = []
    print(
        f"[Sensors] Ground height map at {120 / scan_every:g} Hz"
        + (f" | two lidars, {len(rig.rays)} rays each" if args.lidars else "")
        + (f" | RGB-D {', '.join(args.rgb_cameras)} at {120 / camera_every:g} Hz" if cameras else "")
    )
    while sim.is_running() and (args.max_steps < 0 or step < args.max_steps):
        commands = teleop.axes()
        if args.auto_dig:
            commands = scripted_commands(step / 120, args.auto_cycles, targets, default, joint_ids)
        if args.task == "dig":  # a full load held up with the arm still, the expert done, or the time out ends it
            still = np.abs(numpy(robot.data.joint_vel)[0, arm_ids]).max() < math.radians(STILL_DEG_S)
            held = held + 1 if still and result()["success"] else 0
            timeout, done = step - started >= DIG_TIMEOUT_S * 120, bool(expert and expert.done)
            if held >= DIG_HOLD_S * 120 or timeout or done:
                ended = "held" if held >= DIG_HOLD_S * 120 else "timeout" if timeout else "expert done"
                print(f"[Dig] {result()} ended: {ended}")
                finished += 1
                if finished == args.episodes:
                    break
                reset()
        if expert:
            if expert.done:
                print(f"[Expert] {result()}")
                finished += 1
                if finished == args.episodes:
                    break
                reset()
            commands = expert.commands([float(targets[0, joint_ids[name]]) for name in JOINTS])
        if bridge:  # the joints follow /excavator/command; a scripted run publishes its own there
            commands = bridge.commands(commands if args.auto_dig or expert else None)
        if teleop.pressed("B", "R"):
            reset()
        if teleop.pressed("Y", "C"):
            episode()
        for name, command in commands.items():
            j = joint_ids[name]
            targets[0, j] = float(
                np.clip(
                    float(targets[0, j]) + command * math.radians(JOINT_AXES[name][3]) / 120,
                    *limits[j],
                )
            )
        robot.actuators.target_command.set_position_index(value=targets, full_data=True)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 120)
        step += 1
        if bridge:
            bridge.state(step / 120, numpy(robot.data.joint_pos)[0])
        if step % 60 == 0:
            park()
        if step % 2 == 0:
            begin = time.perf_counter()
            active = assets.index(container)
            pose = np.concatenate(
                (numpy(bodies.data.body_link_pos_w)[0, active], numpy(bodies.data.body_link_quat_w)[0, active])
            )
            if container.moved(pose):  # pushed or knocked over: the sand follows
                soil.restamp(container.stamp(soil, pose))
                show_container()
            if args.showcase and cameras is not None:
                cameras.prop_poses[:count] = np.concatenate(
                    (numpy(bodies.data.body_link_pos_w)[0], numpy(bodies.data.body_link_quat_w)[0]), axis=1
                )
            soil.step(bucket_pose(), 1 / 60)
            soil_times.append((time.perf_counter() - begin) * 1000)
            if recorder:
                recorder.tick(soil, bucket_pose(), step / 120, soil_times[-1])
        sim_time = step / 120
        lag = time.monotonic() - wall_start - sim_time
        if not args.no_realtime and lag > 0.25:
            wall_start += lag - 0.25
        render_now = step - last_render >= render_every and (
            args.no_realtime or lag <= 0 or step - last_render >= max(8, render_every)
        )
        scan_now = step % scan_every == 0
        camera_now = cameras is not None and step % camera_every == 0
        frame_start = time.perf_counter()
        if scan_now or camera_now or (sim.is_rendering and render_now):
            surface_start = time.perf_counter()
            triangles = surface.update()
            surface_ms = (time.perf_counter() - surface_start) * 1000
        if scan_now or camera_now:
            cost = update_sensors(scan_now, camera_now, rig, cameras, frames, lidar_points, surface, robot, sim_time)
            camera_times.extend(cost)
            if bridge and camera_now:
                bridge.cameras(sim_time, cameras.rgb_images)
            if bridge and scan_now:
                bridge.heightmap(sim_time, rig.oracle)
        if sim.is_rendering and render_now:
            if drawn and (soil.version != drawn_version or surface.ground_changed):
                mesh.update(*soil.mesh(height=surface.display_height))
                drawn_version = soil.version
            if grains is not None:
                points, colors = soil.grains()
                if args.debug_particles:
                    velocities = soil.velocities[soil.active != 0]
                    hot = (
                        (velocities[:, 2] > 0.5)
                        | (np.linalg.norm(velocities[:, :2], axis=1) > 0.75)
                        | (soil.motion_speed[soil.active != 0] > 2.5)
                    )
                    colors[hot] = (0.95, 0.08, 0.03)
                grains.update(points, colors)
            sim.render()
            if recorder:
                recorder.frame(
                    viewer,
                    sim_time,
                    frame_ms=(time.perf_counter() - frame_start) * 1000,
                    surface_ms=surface_ms,
                    surface_triangles=triangles,
                    soil=soil,
                    surface=surface,
                )
            last_render = step
        if not args.no_realtime and lag < 0:
            time.sleep(min(-lag, 1 / 120))
        if step % 240 == 0:
            error_ml = (soil.volume() - soil.initial_volume) * 1e6
            print(
                f"[Soil] t={sim_time:.1f}s | moving={np.count_nonzero(soil.active)} | "
                f"tick={np.mean(soil_times[-120:]):.2f} ms | volume drift={error_ml:+.3f} mL | "
                f"in container={container.volume(soil) * 1000:.2f} L"
                + (f" | cameras={np.median(camera_times[-60:]):.1f} ms" if camera_times else "")
                + (" | pool full: increase --particles" if soil.saturated else "")
            )
    wp.synchronize_device(sim.device)
    positions = soil.positions[soil.active != 0]
    if not np.isfinite(positions).all():
        raise RuntimeError("Soil simulation produced nonfinite positions.")
    if args.screenshot:
        capture_viewport(viewer, args.screenshot)
        print(f"[Screenshot] {args.screenshot}")
    if bridge:
        bridge.end_episode(step / 120, result())
        bridge.close()
    rig.close()
    if frames:
        frames.close()
    if recorder:
        recorder.close()
    wall = time.monotonic() - run_start
    print(
        f"[Done] {step} steps | volume drift={(soil.volume() - soil.initial_volume) * 1e6:+.3f} mL | "
        f"{wall:.1f} s wall, {step / 120 / wall:.2f}x real time"
    )


def main():
    args = arguments()
    sim_cfg, scene_cfg, soil_cfg = configs(args)
    with launch_simulation(SimpleNamespace(sim=sim_cfg, scene=scene_cfg), args):
        run(args, sim_cfg, scene_cfg, soil_cfg)


if __name__ == "__main__":
    main()
