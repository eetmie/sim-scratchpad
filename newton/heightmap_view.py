"""Kit (omni.ui) window for the height map; same look as ../lidar_demo_script.py. Import after Kit starts."""

import numpy as np
from heightmap import GROUND_Z_M, HEIGHTMAP_HEIGHT_M, HEIGHTMAP_SIZE, HEIGHTMAP_X_M, HEIGHTMAP_Y_M, new_heightmap_memory

import omni.ui as ui


class HeightMapView:
    """A UI-only preview: no scene geometry and no render product."""

    def __init__(self, title="Lidar Height Map (Newton)"):
        self.data = new_heightmap_memory()
        self.selected = None
        self.provider = ui.ByteImageProvider()
        self.window = ui.Window(title, width=460, height=800)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label(f"{HEIGHTMAP_SIZE} x {HEIGHTMAP_SIZE} | world +X up, +Y left", height=22)
                ui.Label(
                    f"X {HEIGHTMAP_X_M[0]:.2f}..{HEIGHTMAP_X_M[1]:.2f} m | "
                    f"Y {HEIGHTMAP_Y_M[0]:.2f}..{HEIGHTMAP_Y_M[1]:.2f} m",
                    height=20,
                )
                ui.Label(
                    f"Cell {(HEIGHTMAP_X_M[1] - HEIGHTMAP_X_M[0]) * 1000 / HEIGHTMAP_SIZE:.1f} x "
                    f"{(HEIGHTMAP_Y_M[1] - HEIGHTMAP_Y_M[0]) * 1000 / HEIGHTMAP_SIZE:.1f} mm | "
                    f"ground Z = {GROUND_Z_M:.3f} m",
                    height=20,
                )
                with ui.HStack(height=384):
                    ui.Spacer(width=ui.Fraction(1))
                    self.image = ui.ImageWithProvider(
                        self.provider,
                        width=384,
                        height=384,
                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                        mouse_pressed_fn=self.inspect_cell,
                    )
                    ui.Spacer(width=ui.Fraction(1))
                ui.Label(
                    f"Black {HEIGHTMAP_HEIGHT_M[0] * 100:.0f} cm | gray 0 | "
                    f"white +{HEIGHTMAP_HEIGHT_M[1] * 100:.0f} cm",
                    height=20,
                )
                ui.Label("Checker = unknown. Blue tint = remembered, not in this scan.", height=20)
                self.status = ui.Label("Waiting for the first scan...", height=36, word_wrap=True)
                self.statistics = ui.Label("No measured cells yet.", height=32, word_wrap=True)
                self.inspection = ui.Label("No cell selected.", height=32, word_wrap=True)
                ui.Label(
                    "Newton BVH ray cast; excavator's own hits removed by shape ID. Click a cell to inspect.",
                    height=30,
                    word_wrap=True,
                )
        self.update(self.data)

    def update(self, data):
        self.data = data
        yy, xx = np.indices(data["valid"].shape)
        checker = np.where((xx // 2 + yy // 2) % 2, 24, 48).astype(np.uint8)
        display = np.where(data["valid"], data["gray_u8"], checker)
        rgba = np.empty((*display.shape, 4), dtype=np.uint8)
        rgba[:, :, :3] = display[:, :, None]
        rgba[:, :, 3] = 255
        stale = data["valid"] & ~data["fresh"]
        rgba[stale, :2] = (rgba[stale, :2] * 0.7).astype(np.uint8)
        # Enlarge on CPU with nearest-neighbor sampling to keep individual cells legible.
        rgba = np.repeat(np.repeat(rgba, 6, axis=0), 6, axis=1)
        self.provider.set_data_array(rgba.ravel(), [rgba.shape[1], rgba.shape[0]])
        measured = data["height_m"][data["valid"]]
        if measured.size:
            clipped = np.count_nonzero((measured < HEIGHTMAP_HEIGHT_M[0]) | (measured > HEIGHTMAP_HEIGHT_M[1]))
            self.statistics.text = (
                f"Coverage {100 * measured.size / data['valid'].size:.1f}% "
                f"({100 * np.count_nonzero(data['fresh']) / data['valid'].size:.1f}% this scan) | "
                f"height {measured.min() * 1000:+.1f}..{measured.max() * 1000:+.1f} mm | "
                f"{clipped} cells beyond brightness scale"
            )
        else:
            self.statistics.text = "Coverage 0%: no returns inside this world-space crop."
        self.update_inspection()

    def inspect_cell(self, x, y, button, modifiers):
        if button != 0:
            return
        column = int((x - self.image.screen_position_x) * HEIGHTMAP_SIZE / max(self.image.computed_width, 1))
        row = int((y - self.image.screen_position_y) * HEIGHTMAP_SIZE / max(self.image.computed_height, 1))
        if 0 <= row < HEIGHTMAP_SIZE and 0 <= column < HEIGHTMAP_SIZE:
            self.selected = (row, column)
            self.update_inspection()

    def update_inspection(self):
        if self.selected is None:
            return
        row, column = self.selected
        x = HEIGHTMAP_X_M[1] - (row + 0.5) * (HEIGHTMAP_X_M[1] - HEIGHTMAP_X_M[0]) / HEIGHTMAP_SIZE
        y = HEIGHTMAP_Y_M[1] - (column + 0.5) * (HEIGHTMAP_Y_M[1] - HEIGHTMAP_Y_M[0]) / HEIGHTMAP_SIZE
        if not self.data["valid"][row, column]:
            reading = "unknown (never seen)"
        elif self.data["fresh"][row, column]:
            reading = f"{self.data['height_m'][row, column] * 1000:+.1f} mm; {self.data['counts'][row, column]} hits"
        else:
            age_s = (self.data["stamp_ns"] - self.data["last_seen_ns"][row, column]) / 1e9
            reading = f"{self.data['height_m'][row, column] * 1000:+.1f} mm; remembered, seen {age_s:.1f} s ago"
        self.inspection.text = f"[{row}, {column}] X={x:.3f}, Y={y:.3f} m: {reading}"

    def set_status(self, text):
        self.status.text = text

    def destroy(self):
        self.window.destroy()
