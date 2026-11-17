# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The sand-moving task: an open container beside the robot, a randomised sand surface, and the instruction."""

from pathlib import Path

import numpy as np

# Household containers from nvidia/simready-assets, each one rigid body no wider than 27 cm and no taller than
# 13 cm: (asset folder, what the instruction calls it). Fetch them from sim-scratchpad with
#   hf download nvidia/simready-assets --type dataset --include "<folder>/*" --exclude "*/Textures/*"
#       --exclude "*/source_assets/*" --exclude "*/.thumbs/*" --local-dir assets/simready
# shortcut: no textures, the sim draws each container in one flat colour (COLORS); fetch them to shade with them.
ROOT = Path(__file__).resolve().parent.parent / "assets/simready"
ASSETS = [
    ("kitchenware_bowl_b01", "bowl"),
    ("kitchenware_bowl_c01", "bowl"),
    ("kitchenware_bowl_d01", "bowl"),
    ("kitchenware_bowl_e01", "bowl"),
    ("kitchenware_bowl_g01", "bowl"),
    ("kitchenware_bowl_h01", "bowl"),
    ("kitchenware_bowl_i01", "bowl"),
    ("kitchenware_bowl_j01", "bowl"),
    ("kitchenware_bowl_k01", "bowl"),
    ("kitchenware_mug_a01", "mug"),
    ("kitchenware_mug_b01", "mug"),
    ("kitchenware_mug_c01", "mug"),
    ("kitchenware_mug_e01", "mug"),  # not mug_d01: it does not stand, it rolls onto its side
    ("dishware_mug_coffee_a01", "coffee mug"),
    ("kitchenware_cup_a01", "cup"),
    ("kitchenware_cup_d01", "cup"),
    ("dishware_cup_tea_a01", "tea cup"),
    ("dishware_cup_paper_a01", "paper cup"),
    ("kitchenware_jug_a01", "jug"),
    ("dishware_plastic_tray_a01", "tray"),
]
COLORS = {  # sRGB; the first is the nominal one
    "white": (0.85, 0.85, 0.82),
    "red": (0.80, 0.10, 0.08),
    "blue": (0.10, 0.25, 0.80),
    "green": (0.10, 0.55, 0.15),
    "yellow": (0.90, 0.78, 0.10),
    "orange": (0.92, 0.45, 0.08),
    "purple": (0.45, 0.15, 0.65),
    "pink": (0.92, 0.45, 0.65),
    "black": (0.05, 0.05, 0.05),
}
INSTRUCTION = "dig the sand and move it to the {colour} {noun}"
# Container centre, polar about the slew axis: distance [m] and angle from straight ahead [deg], either side.
RADIUS_M = (0.45, 0.65)
ANGLE_DEG = (50.0, 105.0)
# Left only (+1; right is -1): one camera cannot see the pit and the container together, so a policy without
# memory has to know which way to slew. Add -1, and the side to INSTRUCTION, to use both.
SIDES = (1,)
NOMINAL = (0.55, 70.0, 0.0)  # distance, angle (positive = left), yaw [rad]
SAND_LEVEL_M = (-0.04, 0.0)  # mean fill level relative to the brim of the pit
SAND_KINDS = ("rough", "pile", "dug")
PIT_CLEARANCE_M = 0.03
# shortcut: an episode counts as a success with this much sand resting in the container [L] (the bucket carries
# about 0.2); tune it once the expert shows what a good stroke delivers.
SUCCESS_L = 0.05
# The digging task (demo.py --task dig): one cut, the load lifted clear of the pit, then a reset. Sand lies only
# in a dig zone in the arm's reach, the rest of the pit is bare, and the terrain decides where and how high to cut:
#   heaps: a thin layer with one to three heaps; only a cut through a heap fills the bucket.
#   big:   one big heap up to about the brim; the cut starts high.
#   dug:   a thicker layer at a random level with old trenches dug toward the bottom; cut beside them, at its level.
DIG_INSTRUCTION = "dig where the sand is deepest and lift a full bucket"
DIG_KINDS = ("heaps", "big", "dug")
DIG_ZONE = ((0.38, 0.70), (-20.0, 20.0))  # distance from the slew axis [m], angle [deg]: the expert's strokes
DIG_LAYER_M = (0.0, 0.01)  # heaps: sand over the hard bottom between them; often bare
DIG_HEAPS = (1, 4)  # heaps: [low, high) per episode, of different sizes
DIG_HEAP_M = (0.06, 0.14)  # heaps: height over the layer
DIG_BIG_M = (0.18, 0.24)  # big: height over the hard bottom (the brim is at 0.15); settles to its slope
DIG_DUG_M = (0.05, 0.11)  # dug: layer thickness
LIFT_M = 0.03  # sand this far over the brim counts as lifted (the expert holds it at 0.10-0.13)
BUCKET_UP_M = 0.10  # and only with the bucket link this far over the brim: clear of any heap (the expert: 0.144)
# Measured (seed 0, 6 episodes): the expert's cut through the highest swath lifts 0.57-0.68 L, a cut through
# the emptiest one 0.07-0.27 L (0.49 where heaps covered the pit). So a full bucket needs the right spot.
DIG_SUCCESS_L = 0.4
DIG_HOLD_S = 0.5  # the episode ends once a full load has been held up, the arm still, this long
SETTLED_MPS = 0.05  # grains slower than this count as held, faster ones are in flight
STILL_DEG_S = 4.0  # every arm joint slower than this counts as holding still (a policy jitters a little)
DIG_TIMEOUT_S = 25.0  # a policy run that has not lifted a load by then is reset as a failure


def load_mesh(usd_path):
    """Visual triangles of a USD asset in its own frame [m]: (points, faces)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    points, faces, offset = [], [], 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputePurpose() not in ("default", "render"):
            continue
        mesh = UsdGeom.Mesh(prim)
        to_asset = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        p = np.array(mesh.GetPointsAttr().Get()) @ to_asset[:3, :3] + to_asset[3, :3]
        ids, start = np.array(mesh.GetFaceVertexIndicesAttr().Get()), 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            for k in range(1, count - 1):  # fan: the asset's faces are convex quads
                faces.append((ids[start] + offset, ids[start + k] + offset, ids[start + k + 1] + offset))
            start += count
        points.append(p)
        offset += len(p)
    if not points:
        raise ValueError(f"No visual mesh in {usd_path}")
    points = np.concatenate(points) * UsdGeom.GetStageMetersPerUnit(stage)
    return points.astype(np.float32), np.asarray(faces, np.int32).ravel()


def containers(count=None):
    """The first ``count`` assets of the list that are on disk (all by default)."""
    found = [Container(path, noun) for folder, noun in ASSETS for path in (ROOT / folder / "simready_usd").glob("*.usd")]
    return found[:count]


def parked(index, ground):
    """Where an unused container waits: in the air, far outside the soil grid and the cameras' view.

    In the air because resting on the ground costs contacts: 20 parked containers halved the frame rate. The
    caller has to put them back here before they fall the 5 m (one second).
    """
    return np.array([20 + 0.6 * index, 20, ground + 5.0, 0, 0, 0, 1], np.float32)


class Container:
    """An open-top asset standing on the ground. The soil sees its top surface as rigid floor."""

    def __init__(self, usd_path, noun="container"):
        self.path, self.noun, self.name = Path(usd_path), noun, Path(usd_path).parents[1].name
        self.points, self.faces = load_mesh(usd_path)
        self.radius = float(np.linalg.norm(self.points[:, :2], axis=1).max())
        self.pose = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)  # xyz + xyzw, like the link poses
        self.reach = float(np.linalg.norm(self.points, axis=1).max())  # of any point, in any orientation
        self.scene = None
        # The ground mesh has a gap around the outline (Soil.reset); a ground-coloured disc just below closes it.
        turn = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        rim = (self.radius + 0.03) * np.column_stack((np.cos(turn), np.sin(turn), np.zeros(24))) - (0, 0, 5e-4)
        self.pad = np.vstack(((0, 0, -5e-4), rim)).astype(np.float32)
        self.pad_faces = np.array([(0, k + 1, (k + 1) % 24 + 1) for k in range(24)], np.int32).ravel()

    def world_points(self):
        from soil import pose_matrix

        return (self.points @ pose_matrix(self.pose).T + self.pose[:3]).astype(np.float32)

    def pad_pose(self, ground):
        """The pad lies flat on the ground under the container, however the container is tipped."""
        return np.array([self.pose[0], self.pose[1], ground, 0, 0, 0, 1], np.float32)

    def moved(self, pose):
        """Whether ``pose`` (xyz + xyzw) differs from the current one by more than 1 mm or about half a degree."""
        return np.abs(pose[:3] - self.pose[:3]).max() > 1e-3 or abs(np.dot(pose[3:], self.pose[3:])) < 1 - 1e-5

    def stamp(self, soil, pose):
        """Take ``pose`` and return the stamp for ``Soil.restamp``: the top surface seen from above, NaN elsewhere."""
        from lidar import TriangleScene

        self.pose[:] = pose
        if self.scene is None:
            self.scene = TriangleScene(soil.device)
        self.scene.update("ground", self.world_points(), self.faces)
        near = np.hypot(soil.cx - pose[0], soil.cy - pose[1]) < self.reach + soil.cfg.cell
        top = float(pose[2]) + 1.0
        starts = np.column_stack((soil.cx[near], soil.cy[near], np.full(near.sum(), top))).astype(np.float32)
        down = np.tile(np.array([0, 0, -1], np.float32), (len(starts), 1))
        distance, hit = self.scene.cast(starts, down, near=0, far=2)
        stamp = np.full(soil.height.shape, np.nan, np.float32)
        stamp[near] = np.where(hit != 0, top - distance, np.nan)
        return stamp

    def volume(self, soil):
        """Resting sand held on or in the container [m^3]."""
        return float(np.sum((soil.height - soil.base)[soil.footprint], dtype=np.float64) * soil.cfg.cell**2)


def random_top(soil, rng):
    """A sand surface for the pit [m] and its kind. It is never level: a random fill level with broad mounds and
    hollows all over (several cm of relief, so where to dig has to be read from the height map), then

    rough: nothing more; pile: one heap, which may rise above the brim;
    dug: one or two trenches like earlier strokes leave, running toward the machine.
    Hollows that would go below the pit's hard bottom leave it bare.
    """
    kind = str(rng.choice(SAND_KINDS))
    top = np.full(soil.height.shape, soil.cfg.ground + rng.uniform(*SAND_LEVEL_M))

    def bump(height, along, across):
        x, y = rng.uniform(0.35, 0.75), rng.uniform(-0.20, 0.20)
        return height * np.exp(-(((soil.cx - x) / along) ** 2 + ((soil.cy - y) / across) ** 2) / 2)

    for _ in range(rng.integers(4, 10)):
        width = rng.uniform(0.05, 0.12)
        top += bump(rng.uniform(-0.04, 0.04), width, width)
    if kind == "pile":
        width = rng.uniform(0.08, 0.13)
        top += bump(rng.uniform(0.04, 0.08), width, width)
    elif kind == "dug":
        for _ in range(rng.integers(1, 3)):
            top += bump(-rng.uniform(0.04, 0.08), rng.uniform(0.10, 0.18), rng.uniform(0.04, 0.06))
    return top, kind


def dig_top(soil, rng, probe=None):
    """A sand surface for the digging task [m] and its kind (``DIG_KINDS``), sand only in ``DIG_ZONE``.

    ``probe`` [deg]: a test layout instead, kind "probe": one heap at that angle (left positive) on the thin layer,
    so only a cut there fills the bucket.
    """
    floor = soil.cfg.floor
    kind = "probe" if probe is not None else str(rng.choice(DIG_KINDS))

    def mound(height, along, across, at=None):  # Gaussian, centred at a random place in the zone unless ``at``
        distance, angle = at or (rng.uniform(0.44, 0.62), np.radians(rng.uniform(-14.0, 14.0)))
        x, y = distance * np.cos(angle), distance * np.sin(angle)
        return height * np.exp(-(((soil.cx - x) / along) ** 2 + ((soil.cy - y) / across) ** 2) / 2)

    if kind in ("heaps", "probe"):
        layer = floor + rng.uniform(*DIG_LAYER_M) + sum(mound(rng.uniform(-0.008, 0.008), 0.06, 0.06) for _ in range(4))
        top = layer
        for _ in range(1 if probe is not None else rng.integers(*DIG_HEAPS)):  # overlapping heaps merge (max)
            size = rng.uniform(0.08, 0.13), rng.uniform(0.07, 0.11)
            at = None if probe is None else (rng.uniform(0.44, 0.62), np.radians(probe))
            top = np.maximum(top, layer + mound(rng.uniform(*DIG_HEAP_M), *size, at=at))
    elif kind == "big":
        top = floor + mound(rng.uniform(*DIG_BIG_M), rng.uniform(0.19, 0.23), rng.uniform(0.19, 0.23))
    else:
        top = floor + rng.uniform(*DIG_DUG_M) + sum(mound(rng.uniform(-0.01, 0.01), 0.07, 0.07) for _ in range(4))
        for _ in range(rng.integers(1, 3)):  # a trench along a stroke line, toward the machine
            angle = np.radians(rng.uniform(-16.0, 16.0))
            top -= mound(rng.uniform(0.04, 0.09), 0.10, 0.04, at=(rng.uniform(0.48, 0.58), angle))
            # shortcut: the trench runs along x, not along its angle; fine within 16 degrees of straight ahead.
    distance, angle = np.hypot(soil.cx, soil.cy), np.degrees(np.arctan2(soil.cy, soil.cx))
    zone = (distance > DIG_ZONE[0][0]) & (distance < DIG_ZONE[0][1]) & (np.abs(angle) < DIG_ZONE[1][1])
    return np.where(zone, np.maximum(top, floor), floor), kind


def lifted(soil):
    """Sand held up clear of the pit [m^3]: grains over the brim at rest in the bucket, not sand thrown up."""
    up = (soil.active != 0) & (soil.positions[:, 2] > soil.cfg.ground + LIFT_M)
    up &= np.linalg.norm(soil.velocities, axis=1) < SETTLED_MPS
    return float(soil.volumes[up].sum())


def new_episode(soil, containers, rng=None, dig=False, probe=None):
    """Reset the sand and stand one container beside the pit: nominal without ``rng``, random with it.

    ``dig``: the digging task's sand (``dig_top``) and instruction; the container then just stands there.

    Returns its index in ``containers`` and the layout drawn, which names the container, its colour and the task.
    """
    radius, angle, yaw = NOMINAL
    top, kind, index, colour = soil.cfg.ground, "level", 0, next(iter(COLORS))
    if rng is not None:
        top, kind = dig_top(soil, rng, probe) if dig else random_top(soil, rng)
        index, colour = int(rng.integers(len(containers))), str(rng.choice(list(COLORS)))
    container = containers[index]
    while rng is not None:
        radius, angle = rng.uniform(*RADIUS_M), rng.uniform(*ANGLE_DEG) * rng.choice(SIDES)
        yaw = rng.uniform(-np.pi, np.pi)
        x, y = radius * np.cos(np.radians(angle)), radius * np.sin(np.radians(angle))
        # Keep the whole container on hard ground, clear of the pit.
        if np.hypot(max(0.25 - x, 0, x - 0.85), max(abs(y) - 0.30, 0)) > container.radius + PIT_CLEARANCE_M:
            break
    xy = radius * np.cos(np.radians(angle)), radius * np.sin(np.radians(angle))
    pose = (*xy, soil.cfg.ground, 0, 0, np.sin(yaw / 2), np.cos(yaw / 2))
    soil.reset(top, stamp=container.stamp(soil, pose))
    return index, {
        "task": DIG_INSTRUCTION if dig else INSTRUCTION.format(colour=colour, noun=container.noun),
        "container": container.name,
        "colour": colour,
        "container_xy_m": [round(float(v), 4) for v in xy],
        "container_yaw_rad": round(float(yaw), 4),
        "sand": kind,
        "sand_volume_l": round(soil.initial_volume * 1000, 3),
        "sand_top_m": [round(float(v), 4) for v in (soil.height[soil.block].min(), soil.height[soil.block].max())],
    }
