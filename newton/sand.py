"""Lightweight diggable sand: a height grid laid on the terrain, plus a sheet-metal bucket.

Not granular physics, just a few rules that look right at demo speed:

* The sand is a thickness grid ``S`` over the terrain height ``B``; its surface is ``B + S``.
* The bucket is its 2D mid-plane profile (a bent sheet: back wall, floor, lip) extruded over its width.
  Sand above the sheet in any grid column the sheet passes through is cut. It goes into the bucket
  while the opening faces up and there is room; otherwise it is bulldozed to the cells around the bucket.
* With the opening facing down, the load pours out under the opening.
* Slopes steeper than the angle of repose slump towards their lower neighbours.

The sand has no physical collision: the bucket passes through it, and the rocks stay buried in it.
The lidar sees it through :func:`SandBed.raycast` (a Warp ray march against the height grid).
"""

import math

import numpy as np
import warp as wp

# Bucket mid-plane profile in the bucket body frame (y, z) [m], traced from excavator_bucket.usd:
# from the top of the back wall down the curve to the floor and the cutting lip.
BUCKET_PROFILE = np.array(
    [
        (0.010, -0.006),
        (-0.004, -0.014),
        (-0.014, -0.024),
        (-0.020, -0.034),
        (-0.020, -0.048),
        (-0.016, -0.061),
        (-0.009, -0.071),
        (0.000, -0.079),
        (0.008, -0.084),
        (0.030, -0.086),
        (0.060, -0.086),
        (0.093, -0.085),
    ]
)
BUCKET_HALF_WIDTH = 0.05  # [m], local x
LOAD_INSET = 0.004  # [m] the drawn load stays this far inside the sheet and side plates
HEAPED = 1.15  # capacity multiplier over the struck (level-full) cavity volume
ABSORB_ABOVE = -0.1  # cut sand enters the bucket while opening_up exceeds this
SPILL_BELOW = -0.15  # load pours out once the opening tips below horizontal
SPILL_RATE = 3.0  # fraction of the load per second at full tilt
REPOSE_DEG = 34.0
SAND_COLOR = (0.76, 0.58, 0.33)


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def polygon_area(points):
    y, z = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))


def clip_below_line(polygon, origin, normal, offset):
    """Part of a 2D polygon where (p - origin) . normal <= offset (Sutherland-Hodgman, one edge)."""
    out = []
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        da, db = np.dot(a - origin, normal) - offset, np.dot(b - origin, normal) - offset
        if da <= 0:
            out.append(a)
        if (da < 0) != (db < 0):
            out.append(a + (b - a) * da / (da - db))
    return np.array(out)


class Bucket:
    """The bucket as a bent sheet (profile x width) and the sand it carries."""

    def __init__(self):
        profile = BUCKET_PROFILE
        self.top, self.lip = profile[0], profile[-1]
        chord = self.lip - self.top
        normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
        centroid = profile.mean(axis=0)
        if np.dot(centroid - self.top, normal) > 0:
            normal = -normal  # point out of the opening
        self.open_normal_2d = normal
        self.open_normal = np.array([0.0, *normal])
        self.depth = -min(np.dot(p - self.top, normal) for p in profile)  # deepest point below the chord
        self.capacity = HEAPED * polygon_area(profile) * 2 * BUCKET_HALF_WIDTH  # [m^3]
        self.load = 0.0

        # Sheet sample points: the profile resampled every 3 mm, across the width, plus both side plates.
        segments = np.diff(profile, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        s = np.linspace(0, lengths.sum(), int(lengths.sum() / 0.003) + 1)
        along = np.column_stack(
            [np.interp(s, np.concatenate([[0], np.cumsum(lengths)]), profile[:, k]) for k in (0, 1)]
        )
        xs = np.linspace(-BUCKET_HALF_WIDTH, BUCKET_HALF_WIDTH, 11)
        sheet = [np.column_stack([np.full(len(along), x), along]) for x in xs]
        ys, zs = np.meshgrid(np.arange(-0.02, 0.095, 0.006), np.arange(-0.086, 0.0, 0.006))
        grid = np.column_stack([ys.ravel(), zs.ravel()])
        inside = grid[[self._inside(p) for p in grid]]
        for x in (-BUCKET_HALF_WIDTH, BUCKET_HALF_WIDTH):
            sheet.append(np.column_stack([np.full(len(inside), x), inside]))
        self.points_local = np.concatenate(sheet)
        self.opening_center = np.array([0.0, *((self.top + self.lip) / 2)])
        # The traced profile is the bucket's outer skin; the drawn load sits LOAD_INSET inside it,
        # so it does not poke through the sheet metal.
        toward = centroid - profile
        self.inner_profile = profile + LOAD_INSET * toward / np.linalg.norm(toward, axis=1, keepdims=True)

    def _inside(self, p):
        """Point-in-polygon test for the cavity (profile closed by the chord)."""
        poly, inside = BUCKET_PROFILE, False
        for i in range(len(poly)):
            a, b = poly[i], poly[i - 1]
            if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
                inside = not inside
        return inside

    def update(self, sand, pos, quat_xyzw, dt):
        """Cut, carry and spill sand for one bucket pose. Returns opening_up (-1..1)."""
        rotation = quat_xyzw_to_matrix(quat_xyzw)
        opening_up = float((rotation @ self.open_normal)[2])
        cut = sand.cut_below(self.points_local @ rotation.T + pos)
        if cut > 0.0:
            room = max(self.capacity - self.load, 0.0) if opening_up > ABSORB_ABOVE else 0.0
            taken = min(cut, room)
            self.load += taken
            sand.bulldoze(cut - taken)
        if opening_up < SPILL_BELOW and self.load > 0.0:
            tilt = min((SPILL_BELOW - opening_up) / 0.4, 1.0)
            poured = self.load * min(SPILL_RATE * tilt * dt, 1.0)
            self.load -= poured
            sand.deposit((rotation @ self.opening_center + pos)[:2], poured)
        return opening_up

    def load_mesh(self, pos, quat_xyzw, load=None):
        """Prism of the filled part of the cavity, in world frame: (points, triangle indices) or None.

        ``load`` overrides the carried volume [m^3] (the 2D slab's grains in the bucket while it runs).
        """
        load = self.load if load is None else load
        if load <= 1e-7:
            return None
        fill = min(load / self.capacity * HEAPED, 1.0)
        section = clip_below_line(self.inner_profile, self.top, self.open_normal_2d, -(1.0 - fill) * self.depth)
        if len(section) < 3:
            return None
        n = len(section)
        center = section.mean(axis=0)
        ring = np.vstack([section, center])  # fan around the centroid
        half_width = BUCKET_HALF_WIDTH - LOAD_INSET
        points = np.vstack([np.column_stack([np.full(n + 1, x), ring]) for x in (-half_width, half_width)])
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces += [(n, j, i), (2 * n + 1, n + 1 + i, n + 1 + j)]  # side plates
            faces += [(i, j, n + 1 + j), (i, n + 1 + j, n + 1 + i)]  # outer band (top surface included)
        rotation = quat_xyzw_to_matrix(quat_xyzw)
        return points @ rotation.T + pos, np.array(faces, dtype=np.int32).ravel()


@wp.kernel
def _ray_heightfield_kernel(
    height: wp.array2d(dtype=wp.float32),
    x0: float,
    y0: float,
    cell: float,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_t: wp.array(dtype=wp.float32),
    step: float,
    out_t: wp.array(dtype=wp.float32),
):
    """First crossing of each ray below the height grid before max_t; -1 if none."""
    i = wp.tid()
    o = origins[i]
    d = directions[i]
    ny = height.shape[0]
    nx = height.shape[1]
    out_t[i] = -1.0
    t_prev = float(0.0)
    t = float(0.0)
    while t <= max_t[i]:
        p = o + t * d
        fx = (p[0] - x0) / cell - 0.5
        fy = (p[1] - y0) / cell - 0.5
        if fx >= 0.0 and fy >= 0.0 and fx < float(nx - 1) and fy < float(ny - 1):
            ix = int(fx)
            iy = int(fy)
            ax = fx - float(ix)
            ay = fy - float(iy)
            h = (1.0 - ay) * ((1.0 - ax) * height[iy, ix] + ax * height[iy, ix + 1]) + ay * (
                (1.0 - ax) * height[iy + 1, ix] + ax * height[iy + 1, ix + 1]
            )
            if p[2] <= h:
                # Linear refinement between the last point above and this one below.
                lo = t_prev
                hi = t
                for _k in range(6):
                    mid = 0.5 * (lo + hi)
                    q = o + mid * d
                    gx = wp.clamp((q[0] - x0) / cell - 0.5, 0.0, float(nx - 1) - 1.0e-3)
                    gy = wp.clamp((q[1] - y0) / cell - 0.5, 0.0, float(ny - 1) - 1.0e-3)
                    jx = int(gx)
                    jy = int(gy)
                    bx = gx - float(jx)
                    by = gy - float(jy)
                    hm = (1.0 - by) * ((1.0 - bx) * height[jy, jx] + bx * height[jy, jx + 1]) + by * (
                        (1.0 - bx) * height[jy + 1, jx] + bx * height[jy + 1, jx + 1]
                    )
                    if q[2] <= hm:
                        hi = mid
                    else:
                        lo = mid
                out_t[i] = hi
                return
        t_prev = t
        t = t + step


class SandBed:
    """Sand thickness on a regular grid over the terrain."""

    def __init__(self, base, x0, y0, cell, device):
        self.base = base.astype(np.float32)  # terrain height per cell [m], shape (ny, nx)
        self.ny, self.nx = base.shape
        self.x0, self.y0, self.cell = x0, y0, cell
        self.device = device
        self.thickness = np.zeros_like(self.base)
        self.slope = math.tan(math.radians(REPOSE_DEG)) * cell
        self.version = 0  # bumps when the surface changes, for renderers
        self._footprint = np.zeros(self.base.shape, dtype=bool)
        iy, ix = np.indices(self.base.shape)
        self.cx = (x0 + (ix + 0.5) * cell).astype(np.float32)
        self.cy = (y0 + (iy + 0.5) * cell).astype(np.float32)
        # Fixed render topology: two triangles per grid cell.
        v = np.arange(self.nx * self.ny).reshape(self.ny, self.nx)
        a, b, c, d = v[:-1, :-1], v[:-1, 1:], v[1:, :-1], v[1:, 1:]
        self.indices = np.stack([a, b, d, a, d, c], axis=-1).astype(np.int32).ravel()
        self._height_wp = None

    # -- shaping ---------------------------------------------------------------------------------------
    def fill_bed(self, x_range, y_range, level, taper, mounds=()):
        """Fill a rectangle up to ``level`` (soft edges over ``taper`` m), plus Gaussian mounds (x, y, h, sigma)."""
        dx = np.minimum(self.cx - x_range[0], x_range[1] - self.cx)
        dy = np.minimum(self.cy - y_range[0], y_range[1] - self.cy)
        edge = np.clip(np.minimum(dx, dy) / taper, 0.0, 1.0)
        surface = level + 0.0 * self.cx
        for x, y, h, sigma in mounds:
            surface = surface + h * np.exp(-((self.cx - x) ** 2 + (self.cy - y) ** 2) / (2 * sigma**2))
        self.thickness = (np.maximum(surface - self.base, 0.0) * edge**2).astype(np.float32)
        self.relax(200)

    def surface(self):
        return self.base + self.thickness

    def render_height(self):
        """Surface for drawing and ray casting; bare cells sink just under the terrain so it shows through."""
        return np.where(self.thickness > 2e-4, self.base + self.thickness, self.base - 0.01).astype(np.float32)

    def _cells(self, xy):
        ix = np.floor((xy[:, 0] - self.x0) / self.cell).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.y0) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy[ok] * self.nx + ix[ok], ok

    # -- bucket interaction -----------------------------------------------------------------------------
    def cut_below(self, points):
        """Remove sand above the lowest bucket point in every column it passes through. Returns volume [m^3]."""
        self._footprint[:] = False
        cells, ok = self._cells(points)
        if not len(cells):
            return 0.0
        low = np.full(self.nx * self.ny, np.inf, dtype=np.float32)
        np.minimum.at(low, cells, points[ok, 2])
        touched = np.unique(cells)
        self._footprint.ravel()[touched] = True
        thickness, base = self.thickness.ravel(), self.base.ravel()
        new = np.clip(low[touched] - base[touched], 0.0, None)
        removed = np.clip(thickness[touched] - new, 0.0, None)
        thickness[touched] -= removed
        volume = float(removed.sum()) * self.cell**2
        if volume > 0.0:
            self.version += 1
        return volume

    def bulldoze(self, volume):
        """Put sand the bucket pushed (could not take) on the cells just around its footprint."""
        if volume <= 0.0 or not self._footprint.any():
            return
        ring = self._footprint.copy()
        for _ in range(2):
            grown = ring.copy()
            grown[1:] |= ring[:-1]
            grown[:-1] |= ring[1:]
            grown[:, 1:] |= ring[:, :-1]
            grown[:, :-1] |= ring[:, 1:]
            ring = grown
        ring &= ~self._footprint
        if ring.any():
            self.thickness[ring] += volume / (ring.sum() * self.cell**2)
            self.version += 1

    def deposit(self, xy, volume, radius=0.025):
        """Pour ``volume`` m^3 in a Gaussian heap centred on ``xy``; sand poured off the grid is lost."""
        if volume <= 0.0:
            return
        weights = np.exp(-((self.cx - xy[0]) ** 2 + (self.cy - xy[1]) ** 2) / (2 * radius**2))
        weights[weights < 1e-3] = 0.0
        total = weights.sum()
        if total > 0.0:
            self.thickness += (volume / self.cell**2) * weights / total
            self.version += 1

    def relax(self, iterations=2, rate=0.2):
        """Angle-of-repose slumping: move sand down slopes steeper than REPOSE_DEG."""
        moved = False
        for _ in range(iterations):
            height = self.base + self.thickness
            budget = 0.25 * self.thickness
            flux_in = np.zeros_like(height)
            flux_out = np.zeros_like(height)
            for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
                neighbour = np.roll(height, -shift, axis=axis)
                flow = np.clip(rate * (height - neighbour - self.slope), 0.0, None)
                flow = np.minimum(flow, budget)
                # No flow across the grid border (np.roll wraps around).
                edge = [slice(None), slice(None)]
                edge[axis] = -1 if shift == 1 else 0
                flow[tuple(edge)] = 0.0
                flux_out += flow
                flux_in += np.roll(flow, shift, axis=axis)
            if flux_out.max(initial=0.0) > 1e-6:
                moved = True
            self.thickness = np.clip(self.thickness - flux_out + flux_in, 0.0, None)
        if moved:
            self.version += 1

    def volume(self):
        return float(self.thickness.sum()) * self.cell**2

    # -- sensing and drawing ----------------------------------------------------------------------------
    def raycast(self, origins, directions, max_t, step=0.003):
        """Distance along each ray to the sand surface, -1 where it is not reached before max_t."""
        height = wp.array(self.render_height(), dtype=wp.float32, device=self.device)
        out = wp.empty(len(origins), dtype=wp.float32, device=self.device)
        wp.launch(
            _ray_heightfield_kernel,
            dim=len(origins),
            inputs=[
                height,
                float(self.x0),
                float(self.y0),
                float(self.cell),
                wp.array(origins.astype(np.float32), dtype=wp.vec3, device=self.device),
                wp.array(directions.astype(np.float32), dtype=wp.vec3, device=self.device),
                wp.array(max_t.astype(np.float32), dtype=wp.float32, device=self.device),
                float(step),
            ],
            outputs=[out],
            device=self.device,
        )
        return out.numpy()

    def mesh(self):
        """Render mesh: (points (N, 3), triangle indices, vertex normals)."""
        height = self.render_height()
        gy, gx = np.gradient(height, self.cell)
        normals = np.dstack([-gx, -gy, np.ones_like(height)])
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        points = np.dstack([self.cx, self.cy, height]).reshape(-1, 3)
        return points.astype(np.float32), self.indices, normals.reshape(-1, 3).astype(np.float32)
