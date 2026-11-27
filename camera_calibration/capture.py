"""Capture monochrome reference frames from the excavator USD cameras in Isaac Sim."""

import argparse
import json
import traceback
from pathlib import Path

from . import BASE_STAGE


def ir_image(rgba, info, *, allow_flat=False):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    if gray.min() == gray.max():
        if allow_flat:
            return gray
        raise RuntimeError("Renderer returned a flat IR image")
    if info["image_rotation_deg"] == 180:
        gray = cv2.rotate(gray, cv2.ROTATE_180)
    if response := info.get("luma_response"):
        floor, gain, gamma, dx, dy, radial = response
        height, width = gray.shape
        x = np.linspace(-1, 1, width, dtype=np.float32)
        y = np.linspace(-1, 1, height, dtype=np.float32)
        shading = np.exp(dy * y + radial * y**2)[:, None] * np.exp(gain + dx * x + radial * x**2)
        lut = np.linspace(0, 1, 256, dtype=np.float32) ** gamma
        gray = np.rint(255 * np.clip(floor + shading * lut[gray], 0, 1)).astype(np.uint8)
    return gray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=BASE_STAGE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera", choices=("cabin", "arm"), action="append")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "disable_viewport_updates": True, "renderer": "RealTimePathTracing"})
    try:
        capture(args.stage.resolve(), args.out, args.camera or ("cabin", "arm"))
    except BaseException:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()


def capture(stage_path: Path, out: Path, names: list[str] | tuple[str, ...]) -> None:
    import cv2
    import numpy as np

    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd

    from .usd_cameras import cameras, describe

    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"Could not open {stage_path}")
    stage = context.get_stage()
    rig = cameras(stage)
    rep.orchestrator.set_capture_on_play(False)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name in names:
            camera = rig[name]
            info = describe(camera)
            width, height = info["intrinsics"]["width"], info["intrinsics"]["height"]
            product = rep.create.render_product(str(camera.GetPath()), (width, height), force_new=True)
            color = rep.annotators.get("LdrColor", device="cpu")
            color.attach([product.path])
            try:
                rep.orchestrator.step(rt_subframes=16, delta_time=0.0)
                rgba = np.asarray(color.get_data())
                if rgba.shape != (height, width, 4) or rgba.dtype != np.uint8:
                    raise RuntimeError(f"{name}: unexpected LdrColor buffer {rgba.shape}, {rgba.dtype}")
                image = ir_image(rgba, info)
                if not cv2.imwrite(str(out / f"{name}.png"), image):
                    raise OSError(f"Could not write {out / f'{name}.png'}")
                (out / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
                print(f"{name}: {width}x{height} mono8 -> {out / f'{name}.png'}", flush=True)
            finally:
                color.detach([product.path])
                product.destroy()


if __name__ == "__main__":
    main()
