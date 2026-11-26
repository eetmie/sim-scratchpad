"""Render timestamped, paused USD poses to the recorder's IR pixel orientation."""

import argparse
import json
import traceback
from pathlib import Path

from . import BASE_STAGE
from .capture import ir_image


def render(stage_path, plan, out, subframes):
    import cv2
    import numpy as np

    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    from .usd_cameras import cameras, describe

    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"Could not open {stage_path}")
    stage = context.get_stage()
    rig = cameras(stage)
    rep.orchestrator.set_capture_on_play(False)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        ops = {}
        for path in plan.get("hide", []):
            UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
        for name, frames in plan["views"].items():
            camera = rig[name]
            info = describe(camera)
            width, height = (info["intrinsics"][k] for k in ("width", "height"))
            product = rep.create.render_product(str(camera.GetPath()), (width, height), force_new=True)
            color = rep.annotators.get("LdrColor", device="cpu")
            color.attach([product.path])
            folder = out / name
            folder.mkdir(exist_ok=True)
            try:
                for number, frame in enumerate(frames):
                    for path, matrix in frame.get("transforms", {}).items():
                        if path not in ops:
                            xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
                            op = xform.AddTransformOp(opSuffix="irReplay")
                            xform.SetXformOpOrder([op], xform.GetResetXformStack())
                            ops[path] = op
                        ops[path].Set(Gf.Matrix4d(*np.asarray(matrix).ravel()))
                    rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0)
                    rgba = np.asarray(color.get_data())
                    if rgba.shape != (height, width, 4) or rgba.dtype != np.uint8:
                        raise RuntimeError(f"{name}: unexpected LdrColor buffer {rgba.shape}, {rgba.dtype}")
                    gray = ir_image(rgba, info)
                    path = folder / f"{frame['index']:06d}.png"
                    if not cv2.imwrite(str(path), gray):
                        raise OSError(f"Could not capture {path}")
                    if number % 30 == 0 or number == len(frames) - 1:
                        print(f"{name}: {number + 1}/{len(frames)} -> {path}", flush=True)
            finally:
                color.detach([product.path])
                product.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=BASE_STAGE)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subframes", type=int, default=8)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "disable_viewport_updates": True,
            "renderer": "RealTimePathTracing",
            "extra_args": ["--/rtx/post/aa/op=0"],
        }
    )
    try:
        render(args.stage.resolve(), plan, args.out, args.subframes)
    except BaseException:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()


if __name__ == "__main__":
    main()
