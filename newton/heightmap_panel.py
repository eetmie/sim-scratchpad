"""Height map window for the Newton viewer (Dear ImGui); same picture as heightmap_view.py in Kit."""

import numpy as np
from heightmap import HEIGHTMAP_HEIGHT_M, HEIGHTMAP_SIZE, HEIGHTMAP_X_M, HEIGHTMAP_Y_M, new_heightmap_memory

CELL_PX = 5


class HeightMapPanel:
    """Register :meth:`draw` as a free-floating ImGui window; call :meth:`update` with each new map."""

    def __init__(self, visualizer):
        self.data = new_heightmap_memory()
        self.status = "Waiting for the first scan..."
        self.statistics = ""
        self.selected = None
        self._colors = None
        visualizer.register_ui_callback(self.draw, position="free")

    def update(self, data):
        self.data = data
        yy, xx = np.indices(data["valid"].shape)
        checker = np.where((xx // 2 + yy // 2) % 2, 24, 48)
        gray = np.where(data["valid"], data["gray_u8"], checker).astype(np.uint32)
        blue = gray.copy()
        stale = data["valid"] & ~data["fresh"]
        red_green = np.where(stale, (gray * 7) // 10, gray)
        # ImGui colours are 0xAABBGGRR.
        self._colors = (0xFF << 24) | (blue << 16) | (red_green << 8) | red_green
        measured = data["height_m"][data["valid"]]
        if measured.size:
            self.statistics = (
                f"Coverage {100 * measured.size / data['valid'].size:.1f}% "
                f"({100 * np.count_nonzero(data['fresh']) / data['valid'].size:.1f}% this scan) | "
                f"height {measured.min() * 1000:+.1f}..{measured.max() * 1000:+.1f} mm"
            )

    def set_status(self, text):
        self.status = text

    def draw(self, imgui):
        imgui.set_next_window_size(
            imgui.ImVec2(HEIGHTMAP_SIZE * CELL_PX + 20, HEIGHTMAP_SIZE * CELL_PX + 190), imgui.Cond_.first_use_ever
        )
        imgui.set_next_window_pos(imgui.ImVec2(20, 60), imgui.Cond_.first_use_ever)
        if imgui.begin("Lidar Height Map")[0]:
            imgui.text(
                f"{HEIGHTMAP_SIZE}x{HEIGHTMAP_SIZE} | X {HEIGHTMAP_X_M[0]:.2f}..{HEIGHTMAP_X_M[1]:.2f} m (up) | "
                f"Y {HEIGHTMAP_Y_M[0]:.2f}..{HEIGHTMAP_Y_M[1]:.2f} m (left)"
            )
            origin = imgui.get_cursor_screen_pos()
            draw_list = imgui.get_window_draw_list()
            if self._colors is not None:
                for row in range(HEIGHTMAP_SIZE):
                    y = origin.y + row * CELL_PX
                    colors = self._colors[row]
                    for col in range(HEIGHTMAP_SIZE):
                        x = origin.x + col * CELL_PX
                        draw_list.add_rect_filled(
                            imgui.ImVec2(x, y), imgui.ImVec2(x + CELL_PX, y + CELL_PX), int(colors[col])
                        )
            imgui.dummy(imgui.ImVec2(HEIGHTMAP_SIZE * CELL_PX, HEIGHTMAP_SIZE * CELL_PX))
            if imgui.is_item_clicked():
                mouse = imgui.get_mouse_pos()
                self.selected = (int((mouse.y - origin.y) // CELL_PX), int((mouse.x - origin.x) // CELL_PX))
            imgui.text(
                f"Black {HEIGHTMAP_HEIGHT_M[0] * 100:.0f} cm | gray 0 | white +{HEIGHTMAP_HEIGHT_M[1] * 100:.0f} cm. "
                "Checker = unknown, blue = remembered."
            )
            imgui.text_wrapped(self.status)
            imgui.text_wrapped(self.statistics)
            imgui.text_wrapped(self._inspection())
        imgui.end()

    def _inspection(self):
        if self.selected is None:
            return "Click a cell to inspect it."
        row, column = self.selected
        if not (0 <= row < HEIGHTMAP_SIZE and 0 <= column < HEIGHTMAP_SIZE):
            return "Click a cell to inspect it."
        x = HEIGHTMAP_X_M[1] - (row + 0.5) * (HEIGHTMAP_X_M[1] - HEIGHTMAP_X_M[0]) / HEIGHTMAP_SIZE
        y = HEIGHTMAP_Y_M[1] - (column + 0.5) * (HEIGHTMAP_Y_M[1] - HEIGHTMAP_Y_M[0]) / HEIGHTMAP_SIZE
        if not self.data["valid"][row, column]:
            return f"[{row}, {column}] X={x:.3f}, Y={y:.3f} m: unknown"
        return f"[{row}, {column}] X={x:.3f}, Y={y:.3f} m: {self.data['height_m'][row, column] * 1000:+.1f} mm"
