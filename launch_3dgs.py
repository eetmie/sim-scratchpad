"""Calibrated camera/map demo with pose replay or learned hydraulic gamepad motion."""

import argparse
import asyncio
import bisect
import json
import logging
import math
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


def create_app(headless: bool, *, sensor_only: bool = False):
    """Start Isaac Sim using the desktop application's versioned shader caches."""
    from isaacsim import SimulationApp

    return SimulationApp(
        {
            "headless": headless,
            "disable_viewport_updates": sensor_only,
            "multi_gpu": False,
            "renderer": "RealTimePathTracing",
            "extra_args": [
                "--/rtx/shaderDb/shaderCachePath="
                "${omni_global_cache}/shaders/shadercache/${kit_version_short}/${kit_git_hash}/${config}",
                "--/rtx/shaderDb/driverShaderCachePath="
                "${omni_global_cache}/shaders/nv_shadercache/${kit_version_short}/${kit_git_hash}",
            ],
        }
    )


def ros_arguments(parser):
    parser.add_argument("--ros", action="store_true", help="Publish local robot maps, TF and /clock over ROS 2.")
    parser.add_argument("--ros-cameras", action="store_true", help="Also render/publish calibrated mono8 IR images.")
    parser.add_argument(
        "--ros-depth", action="store_true", help="Also publish raw depth, converted to ROS millimetres."
    )
    parser.add_argument("--ros-cloud-hz", type=float, default=2.0, help="Map cloud rate in data time; 0 disables it.")
    parser.add_argument(
        "--ros-time",
        choices=("auto", "fixed", "wall"),
        default="auto",
        help="auto: preserve replay exposure times, otherwise fixed map cadence; wall: live host cadence.",
    )
    parser.add_argument("--ros-start-delay", type=float, default=3.0, help="ROS discovery grace period before capture.")


def validate_ros_arguments(parser, args):
    if args.ros:
        args.depth_maps = True
        if args.mapping != "robot":
            parser.error("--ros requires the robot mapping backend (FastDEM)")
    elif args.ros_cameras or args.ros_depth or args.ros_time != "auto":
        parser.error("--ros-cameras, --ros-depth and --ros-time require --ros")
    if any(not math.isfinite(v) or v < 0 for v in (args.ros_cloud_hz, args.ros_start_delay)):
        parser.error("ROS cloud frequency and discovery delay must be finite and nonnegative")


def start_ros(args, stage, depth_mapping, replay, app):
    if not args.ros:
        return None, None, None
    from ros2.map_contract import FrameClock
    from ros2.viewer_bridge import ViewerRosBridge
    from ros2.viewer_scene import ViewerScene

    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    if not manager.is_extension_enabled("isaacsim.ros2.bridge"):
        raise RuntimeError("Could not enable Isaac's ROS 2 bridge; check the ROS launch environment")
    app.update()
    clock = FrameClock(
        args.ros_time, depth_mapping.profile["map_hz"], [f["timestamp_ns"] for f in replay] if replay else ()
    )
    bridge = ViewerRosBridge(
        depth_mapping.profile,
        clock_mode=clock.mode,
        cameras=args.ros_cameras,
        depth=args.ros_depth,
        cloud_hz=args.ros_cloud_hz,
    )
    scene = None
    try:
        scene = ViewerScene(stage, cameras=args.ros_cameras)
        logger.info("ROS maps enabled; clock=%s, IR=%s, raw depth=%s", clock.mode, args.ros_cameras, args.ros_depth)
        discovery_deadline = time.monotonic() + args.ros_start_delay
        while app.is_running() and time.monotonic() < discovery_deadline:
            bridge.clock(1_000_000_000)
            bridge.spin()
            app.update()
    except BaseException:
        if scene is not None:
            scene.close()
        bridge.close()
        raise
    return bridge, scene, clock


def parse_arguments(argv: list[str] | None = None):
    from heightmap_config import configuration_arguments
    from robot_mapping import mapping_arguments
    from scene_motion import DEMO_REPLAY

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("perspective", "cabin", "arm"), default="perspective")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Close after this many frames; 0 keeps the window open.")
    parser.add_argument("--snapshot", type=Path, help="Save the initial viewport as a PNG after rendering 120 frames.")
    parser.add_argument("--depth-maps", action="store_true", help="Show D435i depth and terrain-map panels.")
    configuration_arguments(parser)
    parser.add_argument("--depth-out", type=Path, default=Path("logs/depth_mapping"))
    parser.add_argument("--ground-z", type=float, help="Override the profile's ground reference in stage metres.")
    parser.add_argument(
        "--mode", choices=("replay", "gamepad"), help="Camera/map demo: exact pose replay or gamepad valve control."
    )
    parser.add_argument("--replay", type=Path, help="Replay file for --mode replay; defaults to bundled motion.")
    parser.add_argument(
        "--nodynamics", action="store_true", help="Gamepad: direct joint speeds instead of arm/slew MLPs."
    )
    parser.add_argument(
        "--actuator-repo", type=Path, help="Isaac-hydraulic-actuator checkout (default: sibling checkout)."
    )
    parser.add_argument("--ir-width", type=int, default=224, help="Demo IR image width; aspect ratio is preserved.")
    parser.add_argument("--show-viewport", action="store_true", help="Enable the full scene viewport for diagnostics.")
    mapping_arguments(parser)
    ros_arguments(parser)
    args = parser.parse_args(argv)
    if args.replay:
        if args.mode == "gamepad":
            parser.error("--replay cannot be used with --mode gamepad")
        args.mode = "replay"
    if args.mode:
        args.depth_maps = True
    if args.mode == "replay":
        args.replay = args.replay or DEMO_REPLAY
    if args.mode == "gamepad" and args.ros:
        parser.error("Gamepad mode is currently non-ROS; use --mode replay with --ros")
    if args.nodynamics and args.mode != "gamepad":
        parser.error("--nodynamics applies to --mode gamepad")
    if args.ir_width < 16:
        parser.error("--ir-width must be at least 16")
    if args.mode and not args.ros and args.snapshot and not args.show_viewport:
        parser.error("--snapshot captures the scene viewport; add --show-viewport, or use exported ir_*.png")
    validate_ros_arguments(parser, args)
    if args.frames < 0 or (args.headless and not args.frames):
        parser.error("--headless requires --frames > 0; --frames must not be negative")
    return args


def save_snapshot(app, viewport, path):
    from omni.kit.viewport.utility import capture_viewport_to_file

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    capture = capture_viewport_to_file(viewport, str(path))
    pending = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
    deadline = time.monotonic() + 60
    while (not pending.done() or not path.is_file()) and app.is_running() and time.monotonic() < deadline:
        app.update()
    if not pending.done():
        pending.cancel()
        raise TimeoutError("Viewport capture did not finish within 60 seconds")
    pending.result()
    if not path.is_file():
        raise RuntimeError(f"Viewport capture did not write {path}")
    logger.info("Saved viewport to %s", path)


def create_motion(args, stage, command_source):
    from hydraulic_motion import Gamepad, HydraulicMotion, MotionWorker, SceneJoints

    joints = SceneJoints(stage)
    motion = HydraulicMotion(joints.initial, joints.limits, dynamics=not args.nodynamics, repo=args.actuator_repo)
    worker = MotionWorker(motion)
    joints.apply(motion.position)
    if command_source is None:
        if args.headless:

            def command_source():
                return [0.0] * 4, False
        else:
            gamepad = Gamepad()
            command_source = gamepad.read
            logger.info("%s", gamepad.source.description)
    logger.info("Gamepad: left X slew, left Y tilt, right Y lift, right X bucket; B/R resets.")
    logger.info("Dynamics: %s; fixed 100 Hz CPU control", "direct speeds" if args.nodynamics else "arm + slew MLP")
    return joints, worker, command_source


def configure_display(args, stage, sensor_only):
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.window import get_viewport_window_instances

    viewport = get_active_viewport()
    if viewport is None and not sensor_only:
        raise RuntimeError("Isaac Sim did not create a viewport")
    if sensor_only:
        for window in get_viewport_window_instances():
            window.viewport_api.updates_enabled = False
            window.visible = False
    if viewport is not None and args.camera != "perspective":
        from camera_calibration.usd_cameras import cameras

        viewport.camera_path = cameras(stage)[args.camera].GetPath()
    return viewport


def main(argv: list[str] | None = None, *, command_source=None) -> None:
    from robot_mapping import fastdem_binary, robot_checkout
    from scene_motion import apply_transforms, load_replay

    args = parse_arguments(argv)
    if args.depth_maps and args.mapping == "robot":
        args.robot_repo, args.fastdem = robot_checkout(args.robot_repo), fastdem_binary(args.fastdem)
    replay = load_replay(args.replay) if args.replay else None
    if replay:
        replay_times = [(f["timestamp_ns"] - replay[0]["timestamp_ns"]) * 1e-9 for f in replay]
        replay_duration = replay_times[-1] + (replay_times[1] if len(replay) > 1 else 0.2)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    root = Path(__file__).resolve().parent
    stage_path = root / "demo_env.usd"
    splat_path = root / "assets/3dgs_40k_1m_sh3.usdz"
    if not stage_path.is_file() or not splat_path.is_file():
        raise FileNotFoundError("Expected demo_env.usd and assets/3dgs_40k_1m_sh3.usdz in this checkout")
    with splat_path.open("rb") as asset:
        if asset.read(4) != b"PK\x03\x04":
            raise ValueError("Room scan is not a USDZ archive. Fetch it with git lfs pull.")

    sensor_only = bool(args.mode and not args.ros and not args.show_viewport)
    app = create_app(args.headless, sensor_only=sensor_only)
    depth_mapping = None
    ros_bridge = ros_scene = ros_clock = None
    worker = None
    frame_ms = deque(maxlen=10000)
    report_due = 0
    exit_code = 0
    try:
        import omni.usd

        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"Could not open {stage_path}")
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        splats = [prim for prim in stage.Traverse() if prim.GetTypeName() == "ParticleField3DGaussianSplat"]
        if not splats:
            raise RuntimeError("The room's Gaussian splat payload did not load")
        app.reset_render_settings()
        app.update()
        viewport = configure_display(args, stage, sensor_only)

        if args.depth_maps:
            from depth_mapping import DepthMapping

            depth_mapping = DepthMapping(
                stage,
                args.dataset,
                args.depth_out,
                args.ground_z,
                panel=not args.headless,
                ir_width=args.ir_width if args.mode and not args.ros else None,
                robot_repo=args.robot_repo,
                fastdem=args.fastdem,
                mapping=args.mapping,
                config=args.config,
            )
            logger.info("D435i depth and terrain maps enabled; save directory: %s", args.depth_out.resolve())

        if args.mode == "gamepad":
            joints, worker, command_source = create_motion(args, stage, command_source)

        ros_bridge, ros_scene, ros_clock = start_ros(args, stage, depth_mapping, replay, app)

        logger.info("Opened %s; Gaussian splats: %s", stage_path, [str(prim.GetPath()) for prim in splats])
        logger.info(
            "Display: %s. Close the Isaac Sim window to exit.",
            "cabin/boom IR + terrain maps; viewport disabled" if sensor_only else str(viewport.camera_path),
        )
        frame = 0
        ros_sample = 0
        replay_started, previous_loop, replay_ops = time.monotonic(), 0, {}
        while app.is_running() and (not args.frames or frame < args.frames):
            frame_started = time.perf_counter()
            if worker is not None:
                import omni.timeline

                if omni.timeline.get_timeline_interface().is_playing():
                    raise RuntimeError("Gamepad motion owns USD poses; keep the physics timeline paused")
                command, reset = command_source()
                joints.apply(worker.sample(command, reset))
                if reset:
                    depth_mapping.clear()
            if ros_bridge is not None:
                import omni.timeline

                if omni.timeline.get_timeline_interface().is_playing():
                    raise RuntimeError(
                        "ROS viewer reads paused USD/replay poses. Keep physics paused for consistent TF."
                    )
                ros_bridge.spin()
            if replay:
                if ros_clock is not None and ros_clock.mode == "replay":
                    loop, index = divmod(ros_sample, len(replay))
                else:
                    elapsed = time.monotonic() - replay_started
                    loop = int(elapsed / replay_duration)
                    index = max(0, bisect.bisect_right(replay_times, elapsed % replay_duration) - 1)
                if loop != previous_loop and depth_mapping is not None:
                    depth_mapping.clear()
                previous_loop = loop
                apply_transforms(stage, replay[index]["transforms"], replay_ops)
            frame += 1
            # A depth capture already pumps Kit and renders through Replicator.
            # Service the ordinary viewport only when no capture was due.
            captured = depth_mapping is not None and depth_mapping.update()
            if not captured:
                app.update()
            elif ros_bridge is not None:
                arrays = depth_mapping.latest
                stamp_ns = ros_clock.stamp(ros_sample, int(arrays["capture_host_monotonic_ns"]))
                arrays["ros_stamp_ns"] = stamp_ns
                ros_bridge.publish(
                    arrays,
                    ros_scene.sample(arrays),
                    stamp_ns=stamp_ns,
                    episode=loop if replay else 0,
                    source_ns=replay[index]["timestamp_ns"] if replay else None,
                )
                ros_sample += 1
            if worker is not None and worker.started is None:
                # Exclude shader/camera initialization from the live control timing.
                worker.start()
            frame_ms.append((time.perf_counter() - frame_started) * 1000)
            if worker is not None and time.monotonic() >= report_due:
                report = worker.report()
                inference = report["inference_ms"].get("p95", 0)
                status = (
                    f"Control {report['achieved_hz']:.1f} Hz | MLP p95 {inference:.2f} ms | "
                    f"render loop {1000 / (sum(frame_ms) / len(frame_ms)):.1f} Hz | "
                    f"dropped {report['dropped_wall_s']:.3f} s | clips {report['velocity_clip_steps']}"
                )
                logger.info(status)
                if depth_mapping.window:
                    depth_mapping.control_status.text = status + "\nB / R: reset"
                report_due = time.monotonic() + 5
            if args.snapshot and frame == min(args.frames or 120, 120):
                save_snapshot(app, viewport, args.snapshot)
        if depth_mapping is not None:
            depth_mapping.save()
            logger.info("Saved D435i depth and terrain maps to %s", args.depth_out.resolve())
    except BaseException:
        logger.exception("3DGS viewer failed")
        exit_code = 1
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
                from hydraulic_motion import percentiles

                report = worker.report()
                report.update(
                    render_loop_ms=percentiles(frame_ms),
                    dynamics=not args.nodynamics,
                    viewport_enabled=not sensor_only,
                    ir_width=args.ir_width,
                )
                args.depth_out.mkdir(parents=True, exist_ok=True)
                (args.depth_out / "control_performance.json").write_text(json.dumps(report, indent=2) + "\n")
                logger.info("Control timing: %s", json.dumps(report))
        finally:
            try:
                try:
                    if ros_scene is not None:
                        ros_scene.close()
                finally:
                    try:
                        if ros_bridge is not None:
                            ros_bridge.close()
                    finally:
                        if depth_mapping is not None:
                            depth_mapping.close()
            finally:
                app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
