"""Granular sand in the arm plane: a 2D MPM slab that takes over from the height grid while digging.

Same idea as "Size Doesn't Matter" (Werner et al., 2026): an excavator digs in the vertical plane of
its arm, so the soil only needs simulating in 2D there. While the swing is still, a slab one bucket
wide along the arm plane is converted from the height grid (sand.py) into ~10k particles and run
with an explicit MLS-MPM sand model (Drucker-Prager plasticity, Klar et al. 2016) in Warp. The
bucket is its sheet-metal profile moving through the slab; the terrain is the slab's floor.

Each sand step the slab's column volumes are written back into the height grid cells it covers, so
the grid mesh, the lidar and the height map keep seeing the dug surface. When the swing moves, the
slab hands back: grains inside the bucket become the bucket's carried load (sand.Bucket), the rest
already lives in the grid. Only the grid mesh and the bucket load prism are drawn; the grains
themselves can be shown for debugging (points spread across the slab width).
"""

import math

import numpy as np
import warp as wp
from matplotlib.path import Path
from sand import BUCKET_HALF_WIDTH, BUCKET_PROFILE, LOAD_INSET, SAND_COLOR, quat_xyzw_to_matrix

DX = 0.005  # MPM grid spacing [m]
SPACING = DX / 2  # particle spacing (2 x 2 per cell)
S_RANGE = (0.05, 1.05)  # slab extent along the arm from the swing axis [m]
Z_RANGE = (-0.17, 0.33)  # slab height range, world z [m]
DENSITY = 1600.0  # [kg/m^3]
YOUNG = 3.0e4  # [Pa] soft, so the explicit step can stay large; sand behaviour comes from the friction
POISSON = 0.3
FRICTION_DEG = 35.0  # internal friction angle
MU_TERRAIN = 0.6  # Coulomb friction against the terrain and the bucket
MU_BUCKET = 0.4
SUBSTEP = 1.0 / 60.0 / 56  # ~0.3 ms, below the CFL limit for YOUNG and DX
HALF_WIDTH = BUCKET_HALF_WIDTH - LOAD_INSET  # slab = the bucket's inner width
COPIES = 3  # each grain is drawn this many times across the width, for a denser look
GRAIN_RADIUS = 0.0018  # drawn radius [m]

_MU = YOUNG / (2 * (1 + POISSON))
_LAMBDA = YOUNG * POISSON / ((1 + POISSON) * (1 - 2 * POISSON))
_SIN_PHI = math.sin(math.radians(FRICTION_DEG))
_ALPHA = math.sqrt(2.0 / 3.0) * 2.0 * _SIN_PHI / (3.0 - _SIN_PHI)


@wp.func
def _coulomb(v: wp.vec2, n: wp.vec2, mu: float):
    """Remove the velocity component into a surface with normal n, with Coulomb friction on the rest."""
    vn = wp.dot(v, n)
    if vn >= 0.0:
        return v
    vt = v - vn * n
    lt = wp.length(vt)
    if lt <= -mu * vn:
        return wp.vec2(0.0, 0.0)
    return vt * (1.0 + mu * vn / lt)


@wp.func
def _terrain_at(terrain: wp.array(dtype=float), s: float, s0: float, inv_dx: float):
    f = wp.clamp((s - s0) * inv_dx, 0.0, float(terrain.shape[0] - 1) - 1.0e-3)
    i = int(f)
    a = f - float(i)
    return (1.0 - a) * terrain[i] + a * terrain[i + 1]


@wp.kernel
def _p2g(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.mat22),
    F: wp.array(dtype=wp.mat22),
    grid_mv: wp.array2d(dtype=wp.vec2),
    grid_m: wp.array2d(dtype=float),
    origin: wp.vec2,
    inv_dx: float,
    dt: float,
    vol: float,
    mass: float,
    mu: float,
    lam: float,
    alpha: float,
):
    p = wp.tid()
    xi = (x[p] - origin) * inv_dx
    base_x = int(xi[0] - 0.5)
    base_y = int(xi[1] - 0.5)
    fx = xi - wp.vec2(float(base_x), float(base_y))
    w0 = 0.5 * wp.cw_mul(wp.vec2(1.5) - fx, wp.vec2(1.5) - fx)
    w1 = wp.vec2(0.75) - wp.cw_mul(fx - wp.vec2(1.0), fx - wp.vec2(1.0))
    w2 = 0.5 * wp.cw_mul(fx - wp.vec2(0.5), fx - wp.vec2(0.5))

    # Deformation update, then Drucker-Prager return mapping in Hencky strain (cohesionless sand).
    Fp = (wp.identity(n=2, dtype=float) + dt * C[p]) * F[p]
    U = wp.mat22()
    sig = wp.vec2()
    V = wp.mat22()
    wp.svd2(Fp, U, sig, V)
    e = wp.vec2(wp.log(wp.max(sig[0], 1.0e-4)), wp.log(wp.max(sig[1], 1.0e-4)))
    tr = e[0] + e[1]
    e_hat = e - wp.vec2(0.5 * tr)
    e_norm = wp.length(e_hat)
    if tr >= 0.0:
        e = wp.vec2(0.0, 0.0)  # pulled apart: no stress
    else:
        dgamma = e_norm + (2.0 * lam + 2.0 * mu) / (2.0 * mu) * tr * alpha
        if dgamma > 0.0:
            e = e - dgamma / e_norm * e_hat
    Fp = U * wp.diag(wp.vec2(wp.exp(e[0]), wp.exp(e[1]))) * wp.transpose(V)
    F[p] = Fp
    # Kirchhoff stress of the Hencky model, in the same rotation.
    tr_e = e[0] + e[1]
    tau = U * wp.diag(wp.vec2(2.0 * mu * e[0] + lam * tr_e, 2.0 * mu * e[1] + lam * tr_e)) * wp.transpose(U)
    affine = (-dt * vol * 4.0 * inv_dx * inv_dx) * tau + mass * C[p]

    for i in range(3):
        for j in range(3):
            wx = w0[0]
            if i == 1:
                wx = w1[0]
            elif i == 2:
                wx = w2[0]
            wy = w0[1]
            if j == 1:
                wy = w1[1]
            elif j == 2:
                wy = w2[1]
            weight = wx * wy
            dpos = (wp.vec2(float(i), float(j)) - fx) / inv_dx
            wp.atomic_add(grid_mv, base_x + i, base_y + j, weight * (mass * v[p] + affine * dpos))
            wp.atomic_add(grid_m, base_x + i, base_y + j, weight * mass)


@wp.kernel
def _grid_update(
    grid_mv: wp.array2d(dtype=wp.vec2),
    grid_m: wp.array2d(dtype=float),
    terrain: wp.array(dtype=float),
    poly: wp.array(dtype=wp.vec2),
    poly_vel: wp.array(dtype=wp.vec2),
    origin: wp.vec2,
    dx: float,
    dt: float,
):
    i, j = wp.tid()
    m = grid_m[i, j]
    if m <= 0.0:
        return
    v = grid_mv[i, j] / m + dt * wp.vec2(0.0, -9.81)
    pos = origin + wp.vec2(float(i), float(j)) * dx
    nx = grid_m.shape[0]
    nz = grid_m.shape[1]

    # Terrain floor (a height profile along the slab).
    if pos[1] < terrain[i]:
        slope = (terrain[wp.min(i + 1, nx - 1)] - terrain[wp.max(i - 1, 0)]) / (2.0 * dx)
        v = _coulomb(v, wp.normalize(wp.vec2(-slope, 1.0)), MU_TERRAIN)

    # Bucket sheet: closest point on the profile polyline, pushed apart within one cell of it.
    best = float(1.0e9)
    closest = wp.vec2()
    sheet_v = wp.vec2()
    sheet_n = wp.vec2()
    for k in range(poly.shape[0] - 1):
        a = poly[k]
        ab = poly[k + 1] - a
        t = wp.clamp(wp.dot(pos - a, ab) / wp.dot(ab, ab), 0.0, 1.0)
        c = a + t * ab
        d = wp.length(pos - c)
        if d < best:
            best = d
            closest = c
            sheet_v = (1.0 - t) * poly_vel[k] + t * poly_vel[k + 1]
            sheet_n = wp.normalize(wp.vec2(-ab[1], ab[0]))
    if best < dx:
        n = sheet_n
        if best > 1.0e-6:
            n = (pos - closest) / best
        v = sheet_v + _coulomb(v - sheet_v, n, MU_BUCKET)

    # Slab ends and lid.
    if (i < 3 and v[0] < 0.0) or (i > nx - 4 and v[0] > 0.0):
        v = wp.vec2(0.0, v[1])
    if (j < 3 and v[1] < 0.0) or (j > nz - 4 and v[1] > 0.0):
        v = wp.vec2(v[0], 0.0)
    grid_mv[i, j] = v


@wp.kernel
def _g2p(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.mat22),
    grid_v: wp.array2d(dtype=wp.vec2),
    terrain: wp.array(dtype=float),
    poly: wp.array(dtype=wp.vec2),
    origin: wp.vec2,
    upper: wp.vec2,
    inv_dx: float,
    dt: float,
):
    p = wp.tid()
    xi = (x[p] - origin) * inv_dx
    base_x = int(xi[0] - 0.5)
    base_y = int(xi[1] - 0.5)
    fx = xi - wp.vec2(float(base_x), float(base_y))
    w0 = 0.5 * wp.cw_mul(wp.vec2(1.5) - fx, wp.vec2(1.5) - fx)
    w1 = wp.vec2(0.75) - wp.cw_mul(fx - wp.vec2(1.0), fx - wp.vec2(1.0))
    w2 = 0.5 * wp.cw_mul(fx - wp.vec2(0.5), fx - wp.vec2(0.5))
    new_v = wp.vec2(0.0, 0.0)
    new_C = wp.mat22(0.0)
    for i in range(3):
        for j in range(3):
            wx = w0[0]
            if i == 1:
                wx = w1[0]
            elif i == 2:
                wx = w2[0]
            wy = w0[1]
            if j == 1:
                wy = w1[1]
            elif j == 2:
                wy = w2[1]
            weight = wx * wy
            g = grid_v[base_x + i, base_y + j]
            dpos = wp.vec2(float(i), float(j)) - fx
            new_v += weight * g
            new_C += (4.0 * inv_dx * weight) * wp.outer(g, dpos)
    v[p] = new_v
    C[p] = new_C
    q = x[p] + dt * new_v

    # Keep grains out of the terrain and the sheet (the grid step alone lets them creep in).
    ground = _terrain_at(terrain, q[0], origin[0], inv_dx)
    if q[1] < ground:
        q = wp.vec2(q[0], ground)
    clearance = 0.5 / inv_dx
    for k in range(poly.shape[0] - 1):
        a = poly[k]
        ab = poly[k + 1] - a
        t = wp.clamp(wp.dot(q - a, ab) / wp.dot(ab, ab), 0.0, 1.0)
        d = q - (a + t * ab)
        dist = wp.length(d)
        if dist < clearance and dist > 1.0e-7:
            q = q + (clearance - dist) / dist * d
    margin = 2.5 / inv_dx
    x[p] = wp.vec2(wp.clamp(q[0], origin[0] + margin, upper[0] - margin), wp.clamp(q[1], origin[1] + margin, upper[1] - margin))


class ArmSlab:
    """2D MPM sand slab along the arm plane, coupled one-way to the bucket (the bucket is kinematic)."""

    def __init__(self, sand, bucket, device):
        self.sand, self.bucket, self.device = sand, bucket, device
        self.nx = int(round((S_RANGE[1] - S_RANGE[0]) / DX)) + 1
        self.nz = int(round((Z_RANGE[1] - Z_RANGE[0]) / DX)) + 1
        self.origin = wp.vec2(S_RANGE[0], Z_RANGE[0])
        self.upper = wp.vec2(S_RANGE[1], Z_RANGE[1])
        self.grid_mv = wp.zeros((self.nx, self.nz), dtype=wp.vec2, device=device)
        self.grid_m = wp.zeros((self.nx, self.nz), dtype=float, device=device)
        self.terrain = wp.zeros(self.nx, dtype=float, device=device)
        self.poly = wp.zeros(len(BUCKET_PROFILE), dtype=wp.vec2, device=device)
        self.poly_vel = wp.zeros(len(BUCKET_PROFILE), dtype=wp.vec2, device=device)
        self.active = False
        self.count = 0
        self._graph = None
        self._prev_poly = None
        self.positions = np.zeros((0, 2), np.float32)  # host copy after each step (s, z)

    # -- frames ----------------------------------------------------------------------------------------
    def _to_slab(self, points_w):
        rel = points_w[:, :2] - self.o
        return np.column_stack([rel @ self.dir, points_w[:, 2]]).astype(np.float32)

    def _profile_slab(self, pos, quat):
        rotation = quat_xyzw_to_matrix(quat)
        local = np.column_stack([np.zeros(len(BUCKET_PROFILE)), BUCKET_PROFILE])
        return self._to_slab(local @ rotation.T + pos)

    def _cells(self):
        """Height-grid cells under the slab, and the slab coordinate of each."""
        sand = self.sand
        rel_x, rel_y = sand.cx - self.o[0], sand.cy - self.o[1]
        s = rel_x * self.dir[0] + rel_y * self.dir[1]
        lat = rel_x * self.side[0] + rel_y * self.side[1] - self.lateral
        mask = (np.abs(lat) < HALF_WIDTH) & (s > S_RANGE[0] + 3 * DX) & (s < S_RANGE[1] - 3 * DX)
        return mask, s[mask]

    # -- hand over ---------------------------------------------------------------------------------------
    def activate(self, carriage_pos, carriage_quat, bucket_pos, bucket_quat):
        """Turn the grid strip under the arm into particles (plus the bucket's load)."""
        heading = quat_xyzw_to_matrix(carriage_quat)[:2, 0]
        self.dir = heading / np.linalg.norm(heading)
        self.side = np.array([-self.dir[1], self.dir[0]])
        self.o = np.asarray(carriage_pos[:2], dtype=np.float64)
        self.lateral = float((bucket_pos[:2] - self.o) @ self.side)
        sand = self.sand

        # Floor and surface profiles along the slab, from the grid (averaged across the slab width).
        self.mask, cell_s = self._cells()
        bins = np.clip(((cell_s - S_RANGE[0]) / DX).astype(int), 0, self.nx - 1)
        node_s = S_RANGE[0] + np.arange(self.nx) * DX
        base_sum = np.bincount(bins, sand.base[self.mask], self.nx)
        top_sum = np.bincount(bins, (sand.base + sand.thickness)[self.mask], self.nx)
        n = np.bincount(bins, minlength=self.nx)
        have = n > 0
        floor = np.interp(node_s, node_s[have], base_sum[have] / n[have])
        top = np.interp(node_s, node_s[have], top_sum[have] / n[have])
        self.terrain.assign(floor.astype(np.float32))

        # Grains: columns every SPACING from the floor to the surface ...
        cols = np.arange(S_RANGE[0] + 4 * DX, S_RANGE[1] - 4 * DX, SPACING)
        bottom, height = np.interp(cols, node_s, floor), np.interp(cols, node_s, top)
        layers = np.floor((height - bottom) / SPACING + 0.5).astype(int).clip(0)
        column = np.repeat(np.arange(len(cols)), layers)
        level = np.arange(len(column)) - np.repeat(np.cumsum(layers) - layers, layers)
        grains = [np.column_stack([cols[column], bottom[column] + (level + 0.5) * SPACING])]
        # ... plus the bucket load, filled from the lowest point of the cavity up.
        profile = self._profile_slab(bucket_pos, bucket_quat)
        if self.bucket.load > 0.0:
            lo, hi = profile.min(axis=0), profile.max(axis=0)
            ss, zz = np.meshgrid(np.arange(lo[0], hi[0], SPACING), np.arange(lo[1], hi[1], SPACING))
            candidates = np.column_stack([ss.ravel(), zz.ravel()])
            inner = Path(profile).contains_points(candidates, radius=-DX)
            candidates = candidates[inner][np.argsort(candidates[inner][:, 1])]
            take = int(self.bucket.load / self.cell_volume())
            grains.append(candidates[:take])
            self.bucket.load = 0.0
        x = np.concatenate(grains).astype(np.float32)
        self.count = len(x)
        rng = np.random.default_rng(0)
        self.lat = rng.uniform(-HALF_WIDTH, HALF_WIDTH, (self.count, COPIES)).astype(np.float32)
        self.shade = rng.uniform(0.75, 1.15, (self.count * COPIES, 1)).astype(np.float32)

        self.x = wp.array(x, dtype=wp.vec2, device=self.device)
        self.v = wp.zeros(self.count, dtype=wp.vec2, device=self.device)
        self.C = wp.zeros(self.count, dtype=wp.mat22, device=self.device)
        self.F = wp.array(np.tile(np.eye(2, dtype=np.float32), (self.count, 1, 1)), dtype=wp.mat22, device=self.device)
        self.poly.assign(profile)
        self.poly_vel.zero_()
        self._prev_poly = profile
        self._graph = None
        self.positions = x
        self.active = True
        self._write_back()

    def deactivate(self):
        """Hand back: grains in the bucket become its load; the rest is already in the grid."""
        if not self.active:
            return
        in_bucket = self._in_bucket()
        self.bucket.load += float(in_bucket.sum()) * self.cell_volume()
        self.active = False
        self.count = 0
        self.positions = np.zeros((0, 2), np.float32)

    def discard(self):
        self.active = False
        self.count = 0
        self.positions = np.zeros((0, 2), np.float32)

    def cell_volume(self):
        """Sand volume one grain stands for [m^3] (its 2D area times the slab width)."""
        return SPACING**2 * 2 * HALF_WIDTH

    # -- stepping --------------------------------------------------------------------------------------------
    def _substeps(self, n):
        inv_dx = 1.0 / DX
        for _ in range(n):
            self.grid_mv.zero_()
            self.grid_m.zero_()
            wp.launch(
                _p2g,
                dim=self.count,
                inputs=[self.x, self.v, self.C, self.F, self.grid_mv, self.grid_m, self.origin, inv_dx, SUBSTEP,
                        SPACING**2, DENSITY * SPACING**2, _MU, _LAMBDA, _ALPHA],
                device=self.device,
            )
            wp.launch(
                _grid_update,
                dim=(self.nx, self.nz),
                inputs=[self.grid_mv, self.grid_m, self.terrain, self.poly, self.poly_vel, self.origin, DX, SUBSTEP],
                device=self.device,
            )
            wp.launch(
                _g2p,
                dim=self.count,
                inputs=[self.x, self.v, self.C, self.grid_mv, self.terrain, self.poly, self.origin, self.upper,
                        inv_dx, SUBSTEP],
                device=self.device,
            )

    def step(self, bucket_pos, bucket_quat, dt):
        """Advance the slab by dt with the bucket at its new pose, then mirror it into the height grid."""
        if not self.active or self.count == 0:
            return
        profile = self._profile_slab(bucket_pos, bucket_quat)
        self.poly_vel.assign((profile - self._prev_poly) / dt)
        self.poly.assign(profile)
        self._prev_poly = profile
        n = max(1, int(round(dt / SUBSTEP)))
        if self._graph is None or self._graph_steps != n:
            if wp.get_device(self.device).is_cuda:
                with wp.ScopedCapture(device=self.device) as capture:
                    self._substeps(n)
                self._graph, self._graph_steps = capture.graph, n
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._substeps(n)
        self.positions = self.x.numpy()
        self._write_back()

    def bucket_volume(self):
        """Sand volume [m^3] of the grains inside the bucket."""
        return float(self._in_bucket().sum()) * self.cell_volume() if self.active else 0.0

    def _in_bucket(self):
        if not len(self.positions):
            return np.zeros(0, bool)
        return Path(self._prev_poly).contains_points(self.positions)

    def _write_back(self):
        """Slab column volumes -> thickness of the grid cells under the slab (grains in the bucket excluded)."""
        free = self.positions[~self._in_bucket()]
        _, cell_s = self._cells()
        cell = self.sand.cell
        bins = ((free[:, 0] - S_RANGE[0]) / cell).astype(int)
        area = np.bincount(bins.clip(0), minlength=int((S_RANGE[1] - S_RANGE[0]) / cell) + 1) * SPACING**2
        cell_bins = ((cell_s - S_RANGE[0]) / cell).astype(int)
        thickness = self.sand.thickness.copy()
        thickness[self.mask] = area[cell_bins] / cell
        self.sand.thickness = thickness.astype(np.float32)
        self.sand.version += 1

    # -- drawing -------------------------------------------------------------------------------------------------
    def grains_world(self):
        """World positions of the drawn grains, (N, 3), and their colours."""
        if not self.active or not len(self.positions):
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        s = np.repeat(self.positions[:, 0], COPIES)
        z = np.repeat(self.positions[:, 1], COPIES)
        lat = self.lat.ravel() + self.lateral
        xy = self.o + s[:, None] * self.dir + lat[:, None] * self.side
        points = np.column_stack([xy, z + GRAIN_RADIUS * 0.5]).astype(np.float32)
        return points, (np.asarray(SAND_COLOR, np.float32) * self.shade).clip(0, 1)
