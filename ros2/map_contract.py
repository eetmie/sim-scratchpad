"""Map, image and time conversions shared by the viewer and offline checks. No ROS or Kit imports."""

import math

import numpy as np

FRONT_FRAME = "excavator_front_map"
SURROUND_FRAME = "excavator_surround_map"
UNKNOWN_MM = -32768


def height_metres(mm):
    height = np.asarray(mm, dtype="<f4") * np.float32(0.001)
    height[np.asarray(mm) == UNKNOWN_MM] = np.nan
    return height


def height_image(mm, limit_m=0.12):
    gray = np.rint(128 + 112 * np.clip(np.asarray(mm, float) / (1000 * limit_m), -1, 1)).astype("u1")
    gray[np.asarray(mm) == UNKNOWN_MM] = 0
    return gray


def grid_points(height, x_m, y_m, cell_m):
    """Cell centres in the map frame: row decreases X, column decreases Y; omit unknowns."""
    rows, cols = np.indices(height.shape)
    points = np.stack((x_m[1] - (rows + 0.5) * cell_m, y_m[1] - (cols + 0.5) * cell_m, height), axis=-1)
    return np.ascontiguousarray(points[np.isfinite(height)], dtype="<f4")


def depth_millimetres(z16, sensor):
    """Recorder Z16 -> ROS 16UC1 millimetres in the physical, unrotated optical frame."""
    pixels = np.asarray(z16)
    if sensor["rotation_deg"] == 180:
        pixels = pixels[::-1, ::-1]
    mm = np.rint(pixels.astype(float) * sensor["depth_scale_m"] * 1000)
    mm[(mm < 1) | (mm > 65535) | ~np.isfinite(mm)] = 0
    return np.ascontiguousarray(mm, dtype="<u2")


def image_calibration(info):
    """Calibration and optical-frame roll for an image in the recorder's pixel orientation."""
    intr = dict(info["intrinsics"])
    intr["coeffs"] = list(intr["coeffs"])
    roll = np.eye(4)
    rotation = info["image_rotation_deg"]
    if rotation == 180:
        intr["cx"] = intr["width"] - 1 - intr["cx"]
        intr["cy"] = intr["height"] - 1 - intr["cy"]
        intr["coeffs"][2:4] = [-v for v in intr["coeffs"][2:4]]
        roll[:3, :3] = np.diag([-1, -1, 1])
    elif rotation != 0:
        raise ValueError("Only 0 and 180 degree camera images are supported")
    return intr, roll


class FrameClock:
    """One ROS timeline; retain source and host clocks separately instead of mixing their epochs.

    A fixed/replay clock advances once per captured observation, independent of rendering speed.
    Replay samples are visited in order and each loop occupies its original duration on this timeline.
    """

    def __init__(self, mode, hz, replay_stamps=()):
        if mode not in ("auto", "fixed", "wall") or not math.isfinite(hz) or hz <= 0:
            raise ValueError("Choose auto/fixed/wall time and a positive finite frequency")
        self.source = [int(t) for t in replay_stamps]
        if any(b <= a for a, b in zip(self.source, self.source[1:])):
            raise ValueError("Replay exposure timestamps must increase strictly")
        self.mode = ("replay" if self.source else "fixed") if mode == "auto" else mode
        if self.mode == "fixed" and self.source:
            self.mode = "replay"
        self.period_ns = round(1e9 / hz)
        self.hz = hz
        steps = np.diff(self.source)
        self.loop_ns = (
            self.source[-1] - self.source[0] + (int(np.median(steps)) if len(steps) else self.period_ns)
            if self.source
            else 0
        )
        self.host_origin = None

    def stamp(self, sample, host_monotonic_ns):
        if self.mode == "replay":
            loop, index = divmod(sample, len(self.source))
            elapsed = loop * self.loop_ns + self.source[index] - self.source[0]
        elif self.mode == "fixed":
            elapsed = round(sample * 1e9 / self.hz)
        else:
            if self.host_origin is None:
                self.host_origin = host_monotonic_ns
            elapsed = host_monotonic_ns - self.host_origin
        return 1_000_000_000 + int(elapsed)  # Nonzero from the first observation; rosbag can start immediately.
