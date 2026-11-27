"""Open the registered room scan in Isaac Sim, with physics paused."""

import argparse
import asyncio
import bisect
import json
import logging
import math
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def create_app(headless: bool):
    """Start Isaac Sim using the desktop application's versioned shader caches."""
    from isaacsim import SimulationApp

    return SimulationApp(
        {
            "headless": headless,
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
    import omni.kit.app

    from ros2.map_contract import FrameClock
    from ros2.viewer_bridge import ViewerRosBridge
    from ros2.viewer_scene import ViewerScene

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


def main(argv: list[str] | None = None) -> None:
    from robot_mapping import fastdem_binary, mapping_arguments, robot_checkout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("perspective", "cabin", "arm"), default="perspective")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Close after this many frames; 0 keeps the window open.")
    parser.add_argument("--snapshot", type=Path, help="Save the initial viewport as a PNG after rendering 120 frames.")
    parser.add_argument("--depth-maps", action="store_true", help="Show D435i depth and terrain-map panels.")
    parser.add_argument(
        "--dataset", type=Path, help="Robot dataset containing meta/heightmaps.json and native sessions."
    )
    parser.add_argument("--depth-out", type=Path, default=Path("logs/depth_mapping"))
    parser.add_argument("--ground-z", type=float, default=-0.0245, help="Registered sand surface in stage metres.")
    parser.add_argument(
        "--replay", type=Path, help="Loop a saved replay.json without regenerating the offline comparison"
    )
    mapping_arguments(parser)
    ros_arguments(parser)
    args = parser.parse_args(argv)
    validate_ros_arguments(parser, args)
    if args.depth_maps and args.dataset is None:
        parser.error("--depth-maps requires --dataset")
    if args.frames < 0 or (args.headless and not args.frames):
        parser.error("--headless requires --frames > 0; --frames must not be negative")
    if args.depth_maps and args.mapping == "robot":
        args.robot_repo, args.fastdem = robot_checkout(args.robot_repo), fastdem_binary(args.fastdem)
    replay = json.loads(args.replay.read_text())["views"]["cabin"] if args.replay else None
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

    app = create_app(args.headless)
    depth_mapping = None
    ros_bridge = ros_scene = ros_clock = None
    exit_code = 0
    try:
        import omni.usd
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

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
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac Sim did not create a viewport")
        if args.camera != "perspective":
            from camera_calibration.usd_cameras import cameras

            viewport.camera_path = cameras(stage)[args.camera].GetPath()

        if args.depth_maps:
            from depth_mapping import DepthMapping

            depth_mapping = DepthMapping(
                stage,
                args.dataset,
                args.depth_out,
                args.ground_z,
                panel=not args.headless,
                robot_repo=args.robot_repo,
                fastdem=args.fastdem,
                mapping=args.mapping,
            )
            logger.info("D435i depth and terrain maps enabled; save directory: %s", args.depth_out.resolve())

        ros_bridge, ros_scene, ros_clock = start_ros(args, stage, depth_mapping, replay, app)

        logger.info("Opened %s; Gaussian splats: %s", stage_path, [str(prim.GetPath()) for prim in splats])
        logger.info("Camera: %s. Physics is paused; close the Isaac Sim window to exit.", viewport.camera_path)
        frame = 0
        ros_sample = 0
        replay_started, previous_loop, replay_ops = time.monotonic(), 0, {}
        while app.is_running() and (not args.frames or frame < args.frames):
            if ros_bridge is not None:
                import omni.timeline

                if omni.timeline.get_timeline_interface().is_playing():
                    raise RuntimeError(
                        "ROS viewer reads paused USD/replay poses. Keep physics paused for consistent TF."
                    )
                ros_bridge.spin()
            if replay:
                from camera_calibration.recorded_poses import apply_transforms

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
            if args.snapshot and frame == min(args.frames or 120, 120):
                args.snapshot = args.snapshot.resolve()
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                capture = capture_viewport_to_file(viewport, str(args.snapshot))
                pending = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
                deadline = time.monotonic() + 60
                # Kit schedules the PNG write separately from the capture future.
                while (
                    (not pending.done() or not args.snapshot.is_file())
                    and app.is_running()
                    and time.monotonic() < deadline
                ):
                    app.update()
                if not pending.done():
                    pending.cancel()
                    raise TimeoutError("Viewport capture did not finish within 60 seconds")
                pending.result()
                if not args.snapshot.is_file():
                    raise RuntimeError(f"Viewport capture did not write {args.snapshot}")
                logger.info("Saved viewport to %s", args.snapshot)
        if depth_mapping is not None:
            depth_mapping.save()
            logger.info("Saved D435i depth and terrain maps to %s", args.depth_out.resolve())
    except BaseException:
        logger.exception("3DGS viewer failed")
        exit_code = 1
        raise
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
