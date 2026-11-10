"""Cabin and arm video feeds: two camera sensors on excavator links, a Kit window, and a frame recorder.

The cameras are ordinary Isaac Lab CameraCfg sensors parented to robot links, so they ride along with the
swing and the stick. Kit's RTX renderer draws them, which also shows the sand mesh and the 2D slab grains
(USD prims). Newton's Warp camera renderer only sees physics shapes, so it would miss the sand.
"""

import json
import os

import numpy as np

# Copied from demo_env.usd (cam_cabin, cam_boom): parent link, position [m] and USD camera orientation
# (x, y, z, w; the camera looks down its -Z with +Y up) in the link frame, focal length [mm].
CAMERAS = {
    "cabin": (
        "upper_carriage",
        (0.12053598482716364, 0.13078053081988514, 0.1945077289640722),
        (-0.30678072168114806, 0.352656317044203, 0.6634356292066332, -0.5842707221424084),
        10.0,
    ),
    "arm": (
        "tilt_boom",
        (0.04948515247158214, 0.0, -0.03343605801471816),
        (0.44977522292347727, -0.44977522292347766, -0.5456209754437016, 0.5456209754437013),
        10.0,
    ),
}


def camera_cfgs(width, height, rate_hz):
    """CameraCfg per feed, keyed ``<name>_cam`` for the scene config."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    aperture = 20.955  # [mm], the default horizontal aperture
    cfgs = {}
    for name, (link, position, orientation, focal_length) in CAMERAS.items():
        cfgs[f"{name}_cam"] = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}/{name}_cam",
            offset=CameraCfg.OffsetCfg(pos=position, rot=orientation, convention="opengl"),
            update_period=1.0 / rate_hz,
            data_types=["rgb"],
            width=width,
            height=height,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=aperture,
                clipping_range=(0.01, 20.0),
            ),
        )
    return cfgs


def rgb_frame(camera):
    """Latest env-0 RGB frame as a (H, W, 3) uint8 array."""
    rgb = camera.data.output["rgb"]
    rgb = getattr(rgb, "torch", rgb)
    return rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8)


class CameraFeedWindow:
    """Kit window with the feeds side by side."""

    def __init__(self, names, width, height):
        import omni.ui as ui

        self.providers = {name: ui.ByteImageProvider() for name in names}
        self.window = ui.Window("Camera feeds", width=2 * width + 40, height=height + 70)
        with self.window.frame:
            with ui.HStack(spacing=8):
                for name, provider in self.providers.items():
                    with ui.VStack(spacing=4):
                        ui.Label(name, height=16)
                        ui.ImageWithProvider(provider, width=width, height=height)

    def update(self, frames):
        for name, rgb in frames.items():
            rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=-1)
            self.providers[name].set_data_array(rgba.ravel(), [rgba.shape[1], rgba.shape[0]])

    def destroy(self):
        self.window.destroy()


class FrameRecorder:
    """Writes each camera frame as a PNG plus one JSON line of robot state per frame set (for VLA datasets)."""

    def __init__(self, directory):
        self.directory = directory
        self.index = 0
        for name in CAMERAS:
            os.makedirs(os.path.join(directory, name), exist_ok=True)
        self.log = open(os.path.join(directory, "steps.jsonl"), "a")
        print(f"[Cameras] Recording to {directory}")

    def write(self, frames, record):
        from PIL import Image

        for name, rgb in frames.items():
            Image.fromarray(rgb).save(os.path.join(self.directory, name, f"{self.index:06d}.png"))
        self.log.write(json.dumps({"frame": self.index, **record}) + "\n")
        self.index += 1

    def close(self):
        self.log.close()
