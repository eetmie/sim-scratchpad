"""Exercise the real arm/slew models with synthetic valves and optional RTX load.

Run from this checkout. --render forwards remaining arguments to launch_3dgs;
without it, compare 60/10/2 Hz input/display polling around the 100 Hz worker.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--frame-delay", type=float, default=0, help="Extra render-thread stall in seconds.")
    parser.add_argument("--out", type=Path, default=Path("logs/gamepad/cpu.json"))
    args, remaining = parser.parse_known_args()
    started = None

    def command():
        nonlocal started
        if started is None:
            started = time.perf_counter()
        t = time.perf_counter() - started
        if args.frame_delay:
            time.sleep(args.frame_delay)
        # Exercise all valve channels with reversals, followed by neutral intervals.
        amplitude = 0.35 if t % 8 < 6 else 0
        return [amplitude * math.sin(t * 0.8 + phase) for phase in (0, 0.6, 1.3, 2.1)], False

    if args.render:
        from launch_3dgs import main as viewer

        viewer(["--mode", "gamepad", *remaining], command_source=command)
        return
    if remaining:
        parser.error(f"Unrecognized arguments: {remaining}")
    from hydraulic_motion import HydraulicMotion, MotionWorker, SceneJoints

    from pxr import Usd

    stage = Usd.Stage.Open(str(Path(__file__).resolve().parents[1] / "demo_env.usd"), load=Usd.Stage.LoadNone)
    stage.Load("/World/excavator_bucket")
    rig = SceneJoints(stage)
    results = []
    for fps in (60, 10, 2):
        worker = MotionWorker(HydraulicMotion(rig.initial, rig.limits))
        started = time.perf_counter()
        worker.start()
        try:
            while time.perf_counter() - started < args.seconds:
                worker.sample(*command())
                time.sleep(1 / fps)
        finally:
            worker.close()
        report = worker.report()
        report["input_poll_hz"] = fps
        results.append(report)
        print(json.dumps(report), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
