# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual soil: resting height field plus a bounded pool of 3D Warp PBD particles.

The bucket is a prescribed, open triangle shell. There is no force feedback to
the machine. Each particle carries an explicit bulk volume; activation and
settling transfer that volume rather than inventing or discarding material.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

# Bucket body coordinates, traced from excavator_bucket.usd (same as ../newton/sand.py).
PROFILE = np.array(
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
    ],
    dtype=np.float32,
)
HALF_WIDTH = 0.05
COLOR = np.array((0.50, 0.36, 0.20), dtype=np.float32)


@dataclass
class SoilConfig:
    cell: float = 0.01
    size: float = 2.4
    floor: float = -0.135  # 15 cm of sand over the pit's hard bottom
    # Level of the hard ground around the sand pit. Equal to ``floor`` gives a free-standing sand block instead.
    ground: float = 0.015
    capacity: int = 10000
    substeps: int = 4
    iterations: int = 4
    friction: float = 0.7
    cohesion: float = 0.0
    repose: float = 34.0
    settle_seconds: float = 0.15
    settle_speed: float = 0.15
    bucket_margin: float = 0.003


def bucket_shell():
    """Return the bucket floor, back and side plates, with its mouth left open."""
    n = len(PROFILE)
    points = np.vstack([np.column_stack([np.full(n, x), PROFILE]) for x in (-HALF_WIDTH, HALF_WIDTH)])
    faces = []
    for i in range(n - 1):
        faces.extend(((i, i + 1, n + i + 1), (i, n + i + 1, n + i)))
    # This traced cavity is convex; a fan triangulates each side plate.
    for i in range(1, n - 1):
        faces.extend(((0, i + 1, i), (n, n + i, n + i + 1)))
    return points.astype(np.float32), np.asarray(faces, np.int32).ravel()


def pose_matrix(pose):
    """Rotation matrix for a Warp/Isaac Lab xyzw pose."""
    return np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)


@wp.func
def terrain_height(
    height: wp.array2d(dtype=float),
    p: wp.vec3,
    origin: float,
    cell: float,
    floor: float,
):
    f = (p[0] - origin) / cell - 0.5
    g = (p[1] - origin) / cell - 0.5
    if f < 0.0 or g < 0.0 or f > float(height.shape[1] - 1) or g > float(height.shape[0] - 1):
        return floor
    f = wp.clamp(f, 0.0, float(height.shape[1] - 1) - 0.001)
    g = wp.clamp(g, 0.0, float(height.shape[0] - 1) - 0.001)
    i, j = int(f), int(g)
    a, b = f - float(i), g - float(j)
    return (1.0 - b) * ((1.0 - a) * height[j, i] + a * height[j, i + 1]) + b * (
        (1.0 - a) * height[j + 1, i] + a * height[j + 1, i + 1]
    )


@wp.kernel
def predict(
    x: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=int),
    dt: float,
    contact_shift: wp.array(dtype=wp.vec3),
    ground_normal: wp.array(dtype=wp.vec3),
    bucket_normal: wp.array(dtype=wp.vec3),
    incoming_velocity: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    old[i] = x[i]
    contact_shift[i] = wp.vec3(0.0)
    ground_normal[i] = wp.vec3(0.0)
    bucket_normal[i] = wp.vec3(0.0)
    if active[i] != 0:
        incoming_velocity[i] = v[i] + wp.vec3(0.0, 0.0, -9.81 * dt)
        x[i] += dt * v[i] + wp.vec3(0.0, 0.0, -9.81 * dt * dt)


@wp.func
def terrain_normal(height: wp.array2d(dtype=float), p: wp.vec3, origin: float, cell: float, floor: float):
    dx = wp.vec3(0.5 * cell, 0.0, 0.0)
    dy = wp.vec3(0.0, 0.5 * cell, 0.0)
    gx = (
        terrain_height(height, p + dx, origin, cell, floor) - terrain_height(height, p - dx, origin, cell, floor)
    ) / cell
    gy = (
        terrain_height(height, p + dy, origin, cell, floor) - terrain_height(height, p - dy, origin, cell, floor)
    ) / cell
    return wp.normalize(wp.vec3(-gx, -gy, 1.0))


@wp.func
def interpolate_pose(a: wp.transform, b: wp.transform, t: float):
    return wp.transform(
        wp.lerp(wp.transform_get_translation(a), wp.transform_get_translation(b), t),
        wp.quat_slerp(wp.transform_get_rotation(a), wp.transform_get_rotation(b), t),
    )


@wp.kernel
def solve_grains(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=int),
    radius: wp.array(dtype=float),
    mu: float,
    cohesion: float,
    max_radius: float,
    delta: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    delta[i] = wp.vec3(0.0)
    if active[i] == 0:
        return
    correction = wp.vec3(0.0)
    count = float(0.0)  # Warp needs a dynamic scalar inside the neighbor loop.
    for j in wp.hash_grid_query(grid, x[i], radius[i] + max_radius * 1.2):
        if j == i or active[j] == 0:
            continue
        separation = x[i] - x[j]
        distance = wp.length(separation)
        rest = radius[i] + radius[j]
        if distance < 1.0e-7 or distance > rest * (1.0 + 0.1 * cohesion):
            continue
        n = separation / distance
        penetration = rest - distance
        if penetration > 0.0:
            displacement = (x[i] - old[i]) - (x[j] - old[j])
            tangent = displacement - wp.dot(displacement, n) * n
            length = wp.length(tangent)
            friction = tangent * wp.min(1.0, mu * penetration / wp.max(length, 1.0e-7))
            correction += 0.5 * (penetration * n - friction)
        else:
            correction += 0.05 * cohesion * penetration * n
        count += 1.0
    delta[i] = 0.85 * correction / wp.max(1.0, 0.25 * count)


@wp.kernel
def apply_contacts(
    x: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=int),
    radius: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
    height: wp.array2d(dtype=float),
    origin: float,
    cell: float,
    floor: float,
    mesh: wp.uint64,
    poses: wp.array(dtype=wp.transform),
    t0: float,
    t1: float,
    mu: float,
    contact_stats: wp.array(dtype=wp.vec3),
    contact_shift: wp.array(dtype=wp.vec3),
    ground_normal: wp.array(dtype=wp.vec3),
    bucket_normal: wp.array(dtype=wp.vec3),
    bucket_point: wp.array(dtype=wp.vec3),
    bucket_margin: float,
):
    i = wp.tid()
    if active[i] == 0:
        return
    previous = interpolate_pose(poses[0], poses[1], t0)
    current = interpolate_pose(poses[0], poses[1], t1)
    p = x[i] + delta[i]
    before_contacts = p
    stats = contact_stats[i]
    stats[2] = wp.max(stats[2], wp.length(delta[i]))
    r = radius[i]
    h = terrain_height(height, p, origin, cell, floor)
    depth = h + r - p[2]
    if depth > 0.0:
        normal = terrain_normal(height, p, origin, cell, floor)
        ground_normal[i] = normal
        push = depth * normal[2]
        stats[0] = wp.max(stats[0], push)
        p += push * normal
        travel = p - old[i]
        tangent = travel - wp.dot(travel, normal) * normal
        p -= tangent * wp.min(1.0, mu * push / wp.max(wp.length(tangent), 1.0e-7))
    # Unsigned shell distance gives a continuous normal around plate edges.
    # Orienting triangle normals using the old point makes the normal flip at
    # curved seams, injecting energy into particles trapped in bucket corners.
    local = wp.transform_point(wp.transform_inverse(current), p)
    local_old = wp.transform_point(wp.transform_inverse(previous), old[i])
    clearance = r + bucket_margin
    # Sweep the center in the moving bucket frame. A point-distance test alone
    # misses a plate when both ends of a substep lie outside its narrow band.
    travel = local - local_old
    distance = wp.length(travel)
    if distance > 1e-7:
        direction = travel / distance
        hit = wp.mesh_query_ray(mesh, local_old, direction, distance)
        if hit.result:
            normal = hit.normal
            if wp.dot(normal, direction) > 0.0:
                normal = -normal
            nearest = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
            penetration = clearance - wp.dot(local - nearest, normal)
            if penetration > 0.0:
                local += penetration * normal
                bucket_normal[i] = wp.transform_vector(current, normal)
                bucket_point[i] = nearest
                stats[1] = wp.max(stats[1], penetration)
    query = wp.mesh_query_point_no_sign(mesh, local, clearance)
    if query.result:
        nearest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        offset = local - nearest
        gap = wp.length(offset)
        if gap > 1e-7 and gap < clearance:
            normal = offset / gap
            bucket_normal[i] = wp.transform_vector(current, normal)
            bucket_point[i] = nearest
            push = clearance - gap
            stats[1] = wp.max(stats[1], push)
            local += push * normal
            travel = local - local_old
            tangent = travel - wp.dot(travel, normal) * normal
            local -= tangent * wp.min(1.0, mu * push / wp.max(wp.length(tangent), 1.0e-7))
            p = wp.transform_point(current, local)
    p = wp.transform_point(current, local)
    x[i] = p
    contact_shift[i] += p - before_contacts
    contact_stats[i] = stats


@wp.func
def inelastic_velocity(velocity: wp.vec3, normal: wp.vec3, friction: float):
    vn = wp.dot(velocity, normal)
    tangent = velocity - vn * normal
    damping = wp.min(1.0, friction * wp.max(-vn, 0.0) / wp.max(wp.length(tangent), 1e-7))
    return (1.0 - damping) * tangent + wp.max(vn, 0.0) * normal


@wp.kernel
def finish(
    x: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=int),
    dt: float,
    raw_speed: wp.array(dtype=float),
    poses: wp.array(dtype=wp.transform),
    t0: float,
    t1: float,
    contact_shift: wp.array(dtype=wp.vec3),
    ground_normal: wp.array(dtype=wp.vec3),
    bucket_normal: wp.array(dtype=wp.vec3),
    bucket_point: wp.array(dtype=wp.vec3),
    mu: float,
    grid: wp.uint64,
    radius: wp.array(dtype=float),
    incoming_velocity: wp.array(dtype=wp.vec3),
    max_radius: float,
):
    i = wp.tid()
    if active[i] != 0:
        # Split external overlap recovery from momentum. Bound granular
        # recovery by the incoming local kinetic energy, so relaxing an initial
        # overlap cannot launch a grain. Coherent bulk flow retains its speed.
        velocity = 0.999 * (x[i] - old[i] - contact_shift[i]) / dt
        energy = wp.length_sq(incoming_velocity[i])
        count = float(1.0)
        for j in wp.hash_grid_query(grid, x[i], radius[i] + 1.1 * max_radius):
            if j != i and active[j] != 0 and wp.length(x[i] - x[j]) < 1.1 * (radius[i] + radius[j]):
                energy += wp.length_sq(incoming_velocity[j])
                count += 1.0
        budget = wp.sqrt(energy / count)
        if wp.length_sq(ground_normal[i]) > 0.0:
            velocity = inelastic_velocity(velocity, ground_normal[i], mu)
        if wp.length_sq(bucket_normal[i]) > 0.0:
            current = interpolate_pose(poses[0], poses[1], t1)
            previous = interpolate_pose(poses[0], poses[1], t0)
            nearest = bucket_point[i]
            surface_velocity = (wp.transform_point(current, nearest) - wp.transform_point(previous, nearest)) / dt
            velocity = surface_velocity + inelastic_velocity(velocity - surface_velocity, bucket_normal[i], mu)
            budget = wp.max(budget, wp.length(surface_velocity))
        speed = wp.length(velocity)
        raw_speed[i] = wp.max(raw_speed[i], speed)
        v[i] = velocity * wp.min(1.0, wp.min(2.0, budget) / wp.max(speed, 1.0e-7))


class Soil:
    """Own the resting bed, particle pool, bucket collider and volume transfers."""

    def __init__(self, config=None, device="cuda:0"):
        self.cfg = config or SoilConfig()
        c = self.cfg
        if c.cell <= 0 or c.size <= c.cell or c.capacity < 1 or c.substeps < 1 or c.iterations < 1:
            raise ValueError("Soil dimensions, capacity, substeps and iterations must be positive.")
        self.device = device
        self.n = int(round(c.size / c.cell))
        self.origin = -self.n * c.cell / 2
        self.yy, self.xx = np.mgrid[: self.n, : self.n]
        self.cx = self.origin + (self.xx + 0.5) * c.cell
        self.cy = self.origin + (self.yy + 0.5) * c.cell
        vertices = np.arange(self.n**2).reshape(self.n, self.n)
        a, b = vertices[:-1, :-1], vertices[:-1, 1:]
        d, e = vertices[1:, :-1], vertices[1:, 1:]
        self.triangles = np.stack((a, b, e, a, e, d), axis=-1).astype(np.int32).reshape(-1, 3)
        self.block = (self.cx > 0.25) & (self.cx < 0.85) & (np.abs(self.cy) < 0.30)
        # The hard floor without any object on it: the bottom of the pit, and the ground around it.
        self.plain = np.where(self.block, self.cfg.floor, self.cfg.ground).astype(np.float32)
        self.height_wp = wp.empty((self.n, self.n), dtype=float, device=device)
        self.x = wp.empty(c.capacity, dtype=wp.vec3, device=device)
        self.old = wp.empty_like(self.x)
        self.v = wp.zeros_like(self.x)
        self.delta = wp.zeros_like(self.x)
        self.active_wp = wp.zeros(c.capacity, dtype=int, device=device)
        self.radius_wp = wp.zeros(c.capacity, dtype=float, device=device)
        self.contact_stats = wp.zeros(c.capacity, dtype=wp.vec3, device=device)
        self.raw_speed = wp.zeros(c.capacity, dtype=float, device=device)
        self.contact_shift = wp.zeros_like(self.x)
        self.ground_normal = wp.zeros_like(self.x)
        self.bucket_normal = wp.zeros_like(self.x)
        self.bucket_point = wp.zeros_like(self.x)
        self.incoming_velocity = wp.zeros_like(self.v)
        self.poses_wp = wp.empty(2, dtype=wp.transform, device=device)
        self.grid = wp.HashGrid(64, 64, 64, device=device)
        self.grid.reserve(c.capacity)
        points, indices = bucket_shell()
        self.shell_points = points
        self.shell = wp.Mesh(
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(indices, dtype=int, device=device),
        )
        self.plan_dt = None
        self.reset()

    def reset(self, top=0.015, stamp=None):
        """Refill the pit up to ``top`` (scalar or per-cell heights [m]) and clear the ground; see ``restamp``."""
        c = self.cfg
        self.base = self.plain.copy()
        self.buried = np.zeros_like(self.base)
        self.footprint = np.zeros(self.base.shape, dtype=bool)
        self.indices = self.triangles.ravel()
        self.height = np.where(self.block, np.maximum(self.base, top), self.base).astype(np.float32)
        if stamp is not None:
            self.restamp(stamp)
        self.positions = np.zeros((c.capacity, 3), np.float32)
        # Park unused slots apart and away from the interaction domain.
        ids = np.arange(c.capacity)
        self.positions[:, 0] = 10 + (ids % 100) * c.cell * 3
        self.positions[:, 1] = 10 + (ids // 100) * c.cell * 3
        self.velocities = np.zeros_like(self.positions)
        self.active = np.zeros(c.capacity, np.int32)
        self.radii = np.zeros(c.capacity, np.float32)
        self.volumes = np.zeros(c.capacity, np.float64)
        self.quiet = np.zeros(c.capacity, np.float32)
        self.motion_speed = np.zeros(c.capacity, np.float32)
        self.previous_pose = None
        self.resting = False
        self.saturated = False
        for _ in range(3000):  # an uneven fill must start at rest, or the map drifts before the robot moves (a big heap: ~1700)
            if not self._relax():
                break
        self._upload_particles()
        self.height_wp.assign(self.height)
        self.initial_volume = self.volume()
        self.version = getattr(self, "version", 0) + 1

    def restamp(self, stamp):
        """Place or move the rigid objects on the ground: per-cell top heights [m], NaN where there is none.

        The sand treats them as hard floor, so an open container holds what falls into it. Sand lying on the
        objects is poured level onto their new shape (and runs off from there if they no longer hold it);
        sand they cover stays buried until they move away.
        """
        thickness = self.height - self.base
        carried = float(np.sum(thickness[self.footprint], dtype=np.float64))
        self.height[self.footprint] = (self.plain + self.buried)[self.footprint]
        covered = np.isfinite(stamp)
        if not covered.any():  # nothing left to carry it: the sand stays where it lay
            self.height[self.footprint] += thickness[self.footprint]
            carried = 0.0
        self.footprint = covered
        self.buried = np.where(covered, self.height - self.plain, 0).astype(np.float32)
        self.base = np.where(covered, np.maximum(stamp, self.height), self.plain).astype(np.float32)
        self.height[covered] = self.base[covered]
        if carried > 0:
            # The level L with sum(max(L - base, 0)) == carried, over the k lowest cells that end up wet.
            lowest = np.sort(self.base[covered].astype(np.float64))
            levels = (carried + np.cumsum(lowest)) / np.arange(1, len(lowest) + 1)
            level = levels[np.flatnonzero(levels >= lowest)[-1]]
            self.height[covered] = np.maximum(self.base[covered], level)
        # The ground mesh cannot follow a thin container wall: drop the triangles that straddle its outline.
        inside = covered.ravel()[self.triangles]
        straddling = inside.any(axis=1) & ~inside.all(axis=1)
        self.indices = np.where(straddling[:, None], self.triangles[:, :1], self.triangles).ravel()
        self.resting = False
        self.version = getattr(self, "version", 0) + 1

    def _upload_particles(self):
        self.x.assign(self.positions)
        self.v.assign(self.velocities)
        self.active_wp.assign(self.active)
        self.radius_wp.assign(self.radii)

    def volume(self):
        """Total bulk volume, including airborne and carried material [m^3]."""
        return float(
            (np.sum(self.height - self.base, dtype=np.float64) + np.sum(self.buried, dtype=np.float64))
            * self.cfg.cell**2
            + self.volumes[self.active != 0].sum()
        )

    def _activate(self, pose):
        c = self.cfg
        self.saturated = False
        current = self.shell_points @ pose_matrix(pose).T + pose[:3]
        previous = current
        if self.previous_pose is not None:
            previous = self.shell_points @ pose_matrix(self.previous_pose).T + self.previous_pose[:3]
        bounds = np.vstack((current, previous))
        pad = max(c.bucket_margin + 0.535 * c.cell, 0.75 * c.cell)
        lo, hi = bounds.min(axis=0) - pad, bounds.max(axis=0) + pad
        region = (
            (self.cx >= lo[0])
            & (self.cx <= hi[0])
            & (self.cy >= lo[1])
            & (self.cy <= hi[1])
            & (self.height > np.maximum(lo[2], self.base) + 1.0e-6)
        )
        ys, xs = np.nonzero(region)
        free = np.flatnonzero(self.active == 0)
        cursor = 0
        # Process columns nearest the tool first if the bounded pool fills up.
        order = np.argsort((self.cx[ys, xs] - pose[0]) ** 2 + (self.cy[ys, xs] - pose[1]) ** 2)
        for k in order:
            iy, ix = ys[k], xs[k]
            bottom = max(float(self.base[iy, ix]), float(lo[2]))
            top = float(self.height[iy, ix])
            if top - bottom < 0.25 * c.cell:
                continue  # retain tiny shavings in the grid, rather than spawning near-zero-radius grains
            count = int(np.ceil((top - bottom) / c.cell))
            if cursor + count > len(free):
                self.saturated = True
                continue  # leave the resting column intact: never lose mass at capacity
            slots = free[cursor : cursor + count]
            cursor += count
            edges = np.linspace(bottom, top, count + 1)
            self.positions[slots] = np.column_stack(
                (
                    np.full(count, self.cx[iy, ix]),
                    np.full(count, self.cy[iy, ix]),
                    (edges[:-1] + edges[1:]) / 2,
                )
            )
            self.velocities[slots] = 0
            self.volumes[slots] = np.diff(edges) * c.cell**2
            self.radii[slots] = 0.535 * np.cbrt(self.volumes[slots])
            self.active[slots], self.quiet[slots] = 1, 0
            self.height[iy, ix] = bottom
        if cursor:
            self.version += 1
        return cursor > 0

    def _settle(self, pose, dt):
        c = self.cfg
        ids = np.flatnonzero(self.active)
        if not len(ids):
            return False
        p = self.positions[ids]
        ix = np.floor((p[:, 0] - self.origin) / c.cell).astype(int)
        iy = np.floor((p[:, 1] - self.origin) / c.cell).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < self.n) & (iy < self.n)
        # Never freeze soil inside/near the bucket, including a stationary load.
        local = (p - pose[:3]) @ pose_matrix(pose)
        near_bucket = (
            (np.abs(local[:, 0]) < HALF_WIDTH + 3 * c.cell)
            & (local[:, 1] > -0.02 - 3 * c.cell)
            & (local[:, 1] < 0.093 + 3 * c.cell)
            & (local[:, 2] > -0.086 - 3 * c.cell)
            & (local[:, 2] < 0.02 + 3 * c.cell)
        )
        slow = np.linalg.norm(self.velocities[ids], axis=1) < c.settle_speed
        eligible = inside & slow & ~near_bucket
        self.quiet[ids] = np.where(eligible, self.quiet[ids] + dt, 0)
        candidates = ids[self.quiet[ids] >= c.settle_seconds]
        merged = False
        for i in candidates[np.argsort(self.positions[candidates, 2])]:
            x, y, z = self.positions[i]
            j, k = int((y - self.origin) / c.cell), int((x - self.origin) / c.cell)
            # Only supported bottom particles merge; airborne particles retain their height.
            # Neighbouring columns count as support: grains wedged against a cut wall rest on them.
            support = self.height[max(j - 1, 0) : j + 2, max(k - 1, 0) : k + 2].max()
            if z - self.radii[i] > support + 0.75 * c.cell:
                continue
            self.height[j, k] += self.volumes[i] / c.cell**2
            self.active[i] = 0
            self.positions[i] = (
                10 + (i % 100) * c.cell * 3,
                10 + (i // 100) * c.cell * 3,
                0,
            )
            self.velocities[i] = 0
            merged = True
        return merged

    def _relax(self):
        """Conservative eight-direction transfers, excluding dynamic particle columns."""
        c = self.cfg
        slope = np.tan(np.deg2rad(c.repose)) * c.cell
        moved = False
        # Bare floor cannot flow: work on the bounding window of the sand, plus one cell to receive it.
        sand = self.height > self.base
        rows, cols = np.flatnonzero(sand.any(axis=1)), np.flatnonzero(sand.any(axis=0))
        if not len(rows):
            return False
        y0, x0 = max(rows[0] - 1, 0), max(cols[0] - 1, 0)
        window = (slice(y0, rows[-1] + 2), slice(x0, cols[-1] + 2))
        height, base = self.height[window], self.base[window]
        ny, nx = height.shape
        occupied = np.zeros_like(height, dtype=bool)
        p = self.positions[self.active != 0]
        ix = np.floor((p[:, 0] - self.origin) / c.cell).astype(int) - x0
        iy = np.floor((p[:, 1] - self.origin) / c.cell).astype(int) - y0
        inside = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
        occupied[iy[inside], ix[inside]] = True
        # A one-cell buffer prevents resting soil flowing into the particle bed
        # and displacing the same material a second time.
        occupied[1:] |= occupied[:-1].copy()
        occupied[:-1] |= occupied[1:].copy()
        occupied[:, 1:] |= occupied[:, :-1].copy()
        occupied[:, :-1] |= occupied[:, 1:].copy()
        # Pairwise transfers are sequential to avoid spending a cell's mass twice.
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = (slice(0, ny - dy), slice(max(0, -dx), nx - max(0, dx)))
            b = (slice(dy, ny), slice(max(0, dx), nx - max(0, -dx)))
            diff = height[a] - height[b]
            excess = np.maximum(np.abs(diff) - float(slope * np.hypot(dx, dy)), 0)
            flux = 0.12 * np.sign(diff) * excess
            flux[occupied[a] | occupied[b]] = 0
            flux = np.clip(
                flux,
                -(height[b] - base[b]) * 0.2,
                (height[a] - base[a]) * 0.2,
            )
            # 0.02 mm: below this the bed only creeps by micrometres per tick, yet never comes to rest.
            if np.max(np.abs(flux), initial=0) > 2e-5:
                height[a] -= flux
                height[b] += flux
                moved = True
        return moved

    def step(self, pose, dt):
        """Advance soil using a bucket pose (xyz + xyzw) at the end of this interval."""
        if dt <= 0:
            return
        pose = np.asarray(pose, dtype=np.float32)
        old_positions = self.positions.copy()
        old_active = self.active != 0
        start = pose if self.previous_pose is None else self.previous_pose
        changed = self._activate(pose)
        self.height_wp.assign(self.height)
        if changed:
            self._upload_particles()
        if self.active.any():
            self.contact_stats.zero_()
            self.raw_speed.zero_()
            self.poses_wp.assign(np.stack((start, pose)))
            # No CUDA graph: replaying one that contains HashGrid.build is corrupted by any other
            # hash-grid user (the surface extraction), which changed results and tripled the tick.
            self._simulate(dt)
            self.positions = self.x.numpy()
            self.velocities = self.v.numpy()
            if self._settle(pose, dt):
                self._upload_particles()
                changed = True
        persistent = old_active & (self.active != 0)
        self.motion_speed.fill(0)
        self.motion_speed[persistent] = (
            np.linalg.norm(self.positions[persistent] - old_positions[persistent], axis=1) / dt
        )
        # A settled bed with no grains and no new deposits has nothing left to flow.
        relaxed = (changed or not self.resting or self.active.any()) and self._relax()
        self.resting = not relaxed and not self.active.any()
        if relaxed or changed:
            self.version += 1
        self.previous_pose = pose.copy()

    def _simulate(self, dt):
        # Packing kernel arguments in Python cost more than the GPU work; record the launches once per dt.
        if self.plan_dt != dt:
            self.plan, self.plan_dt = self._plan(dt), dt
        for run in self.plan:
            run()

    def _plan(self, dt):
        plan = []

        def record(kernel, dim, inputs, device):
            plan.append(wp.launch(kernel, dim, inputs, device=device, record_cmd=True).launch)

        sub_dt = dt / self.cfg.substeps
        for s in range(self.cfg.substeps):
            t0, t1 = s / self.cfg.substeps, (s + 1) / self.cfg.substeps
            record(
                predict,
                self.cfg.capacity,
                [
                    self.x,
                    self.old,
                    self.v,
                    self.active_wp,
                    sub_dt,
                    self.contact_shift,
                    self.ground_normal,
                    self.bucket_normal,
                    self.incoming_velocity,
                ],
                device=self.device,
            )
            plan.append(lambda: self.grid.build(self.x, 2.2 * self.cfg.cell))
            for _ in range(self.cfg.iterations):
                record(
                    solve_grains,
                    self.cfg.capacity,
                    [
                        self.grid.id,
                        self.x,
                        self.old,
                        self.active_wp,
                        self.radius_wp,
                        self.cfg.friction,
                        self.cfg.cohesion,
                        self.cfg.cell * 0.535,
                        self.delta,
                    ],
                    device=self.device,
                )
                record(
                    apply_contacts,
                    self.cfg.capacity,
                    [
                        self.x,
                        self.old,
                        self.active_wp,
                        self.radius_wp,
                        self.delta,
                        self.height_wp,
                        self.origin,
                        self.cfg.cell,
                        self.cfg.floor,
                        self.shell.id,
                        self.poses_wp,
                        t0,
                        t1,
                        self.cfg.friction,
                        self.contact_stats,
                        self.contact_shift,
                        self.ground_normal,
                        self.bucket_normal,
                        self.bucket_point,
                        self.cfg.bucket_margin,
                    ],
                    device=self.device,
                )
            record(
                finish,
                self.cfg.capacity,
                [
                    self.x,
                    self.old,
                    self.v,
                    self.active_wp,
                    sub_dt,
                    self.raw_speed,
                    self.poses_wp,
                    t0,
                    t1,
                    self.contact_shift,
                    self.ground_normal,
                    self.bucket_normal,
                    self.bucket_point,
                    self.cfg.friction,
                    self.grid.id,
                    self.radius_wp,
                    self.incoming_velocity,
                    self.cfg.cell * 0.535,
                ],
                device=self.device,
            )
        return plan

    def render_height(self, height=None):
        """Common render surface used by both the grid mesh and particle reconstruction."""
        # A render-only binomial filter removes single-cell deposition spikes.
        # The volume ledger and collision surface retain the simulation heights.
        height = self.height.copy() if height is None else np.asarray(height, dtype=np.float32).copy()
        inside = height[self.footprint]
        height[self.footprint] = self.cfg.ground  # keep container walls out of the blur, and the blur out of them
        for axis in (0, 1):
            padded = np.pad(height, ((1, 1), (0, 0)) if axis == 0 else ((0, 0), (1, 1)), mode="edge")
            if axis == 0:
                height = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
            else:
                height = 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
        height[self.footprint] = inside
        return height.astype(np.float32)

    def mesh(self, height=None, normals=True):
        """Ground geometry and vertex normals, with fixed topology."""
        height = self.render_height() if height is None else height
        points = np.dstack((self.cx, self.cy, height)).astype(np.float32)
        # Where an object stands with no sand on it, its own mesh is what should be seen. A 1 cm grid cannot
        # hide under thin or steep walls, so those vertices all collapse to one point beneath the object: their
        # triangles vanish, and the ones joining them to sand in the object run down underneath that sand.
        bare = self.footprint & (height - self.base < 1e-4)
        if bare.any():
            under = self.cx[self.footprint].mean(), self.cy[self.footprint].mean(), self.cfg.ground - 1e-3
            points[bare] = under
        if not normals:
            return points.reshape(-1, 3), self.indices, None
        gy, gx = np.gradient(height, self.cfg.cell)
        normals = np.dstack((-gx, -gy, np.ones_like(gx)))
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return (
            points.reshape(-1, 3),
            self.indices,
            normals.reshape(-1, 3).astype(np.float32),
        )

    def grains(self):
        ids = np.flatnonzero(self.active)
        shade = 0.8 + 0.35 * ((ids * 2654435761 % 65536) / 65535)
        return self.positions[ids], (shade[:, None] * COLOR).astype(np.float32)
