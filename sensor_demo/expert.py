"""Scripted operator for data collection: a few strokes where the sand lies highest, each carried over the
container and poured.

It plans from what only the sim knows (the sand heights, the container's pose and size) and drives with the same
rate commands as a gamepad, so a recording of it reads like a recording of an operator. Where it digs follows from
the sand surface alone: the stroke whose swath holds the most sand. After a scoop that swath is lower, so the
next one goes beside it, or deeper in the same place if that is still the highest. The pace is drawn per episode.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from soil import PROFILE

JOINTS = ("revolute_carriage", "revolute_lift", "revolute_tilt", "revolute_tool")
URDF = Path(__file__).resolve().parent.parent / "assets/urdf/excavator.urdf"
# The bucket mouth in the bucket link's (y, z): from the top of the back plate to the cutting edge.
BACK, LIP = (0.0, *PROFILE[0]), (0.0, *PROFILE[-1])
LEVEL, OPEN, DOWN, EMPTY = (math.radians(a) for a in (-176, -80, -96, -55))  # mouth pitch: carrying, reaching, cutting
STROKE_AZIMUTH_DEG = 18.0  # strokes start within this of straight ahead
STROKE_START_M = (0.54, 0.68)  # distance of the cutting edge from the slew axis where it enters the sand
STROKE_LENGTH_M = 0.20
STROKE_DEPTH_M = 0.05  # below the swath's surface, but never into the pit's hard bottom
SWATH_M = (-0.035, 0.0, 0.035)  # across the bucket's width
PACE = (0.7, 1.0)  # fraction of full joint speed, drawn per episode
POUR_BACK_M = 0.005  # the cutting edge pours this far short of the container's centre, toward the machine
POUR_ABOVE_M = 0.04  # cutting edge above the container's top (or the heap in it) while pouring
CARRY_ABOVE_M = 0.13  # and while travelling there: the level bucket hangs 9 cm below its edge


def turn(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


class Arm:
    """Boom, arm and bucket in their vertical plane, from the URDF: where the cutting edge is for given joints."""

    def __init__(self, limits, urdf=URDF):
        """``limits``: (lower, upper) [rad] of lift, tilt and tool."""
        joints = {j.get("name"): j.find("origin") for j in ET.parse(urdf).getroot().findall("joint")}
        self.chain = []
        for name in ("world_fixed", *JOINTS, "fix_tool"):
            xyz, rpy = ([float(v) for v in joints[name].get(k).split()] for k in ("xyz", "rpy"))
            self.chain.append((np.array(xyz), turn("z", rpy[2]) @ turn("y", rpy[1]) @ turn("x", rpy[0])))
        self.limits = np.asarray(limits, float)

    def edge(self, lift, tilt, tool):
        """(distance from the slew axis, height, mouth pitch) of the cutting edge; pitch 180 deg = level, facing up."""
        p, r = np.zeros(3), np.eye(3)
        for (xyz, rotation), angle in zip(self.chain, (0, 0, lift, tilt, tool, 0)):
            p, r = p + r @ xyz, r @ rotation @ turn("y", angle)
        lip, back = p + r @ LIP, p + r @ BACK
        return np.array([lip[0], lip[2], math.atan2(lip[2] - back[2], lip[0] - back[0])])

    def solve(self, distance, height, pitch, near):
        """Joints placing the cutting edge there, closest to ``near``; the nearest reachable otherwise."""

        def miss(q):
            r, z, a = self.edge(*q)
            return r - distance, z - height, 0.1 * math.remainder(a - pitch, 2 * math.pi)

        return least_squares(miss, np.clip(near, *self.limits.T), bounds=self.limits.T).x


class Expert:
    def __init__(self, joint_axes, limits, rng):
        """``joint_axes``: the demo's joint -> (..., speed deg/s at command 1) table; ``limits``: per JOINTS [rad]."""
        self.speed = np.radians([joint_axes[name][3] for name in JOINTS])  # signed: what command +1 does
        self.arm, self.rng = Arm(np.asarray(limits)[1:]), rng
        self.goals, self.scoops, self.done, self.strokes, self.pour = [], 0, True, [], True

    def start(self, soil, container, joints, scoops=1, pour=True):
        """Begin ``scoops`` strokes from ``joints`` (JOINTS order [rad]), returning the arm there at the end.

        ``pour=False``: the digging task, each stroke ends holding the load up and the episode ends there.
        """
        self.soil, self.container, self.home, self.pour = soil, container, np.asarray(joints, float), pour
        self.goals, self.scoops, self.done, self.strokes = [], scoops, False, []
        self.pace = self.rng.uniform(*PACE)

    def sand(self, distance, azimuth, aside=0.0):
        """Resting height at distance/direction from the slew axis, with ``aside`` offset to its left."""
        soil = self.soil
        x = distance * math.cos(azimuth) - aside * math.sin(azimuth)
        y = distance * math.sin(azimuth) + aside * math.cos(azimuth)
        return float(soil.height[int((y - soil.origin) / soil.cfg.cell), int((x - soil.origin) / soil.cfg.cell)])

    def stroke(self):
        """(azimuth, start distance, surface there, surface at the stroke's end) of the swath holding the most sand."""

        def swath(azimuth, start, distances):
            return np.mean([self.sand(d, azimuth, aside) for d in distances for aside in SWATH_M])

        lines = np.radians(np.linspace(-STROKE_AZIMUTH_DEG, STROKE_AZIMUTH_DEG, 13))
        starts = np.linspace(*STROKE_START_M, 8)
        fill = [[swath(a, s, np.linspace(s - STROKE_LENGTH_M, s, 6)) for s in starts] for a in lines]
        i, j = np.unravel_index(np.argmax(fill), (len(lines), len(starts)))
        azimuth, start = lines[i], starts[j]
        return azimuth, start, swath(azimuth, start, [start]), swath(azimuth, start, [start - STROKE_LENGTH_M])

    def plan(self, joints):
        """Queue one stroke and its pour, starting from ``joints``."""
        azimuth, start, far, near = self.stroke()
        bottom = self.soil.cfg.floor + 0.01
        x, y = self.container.pose[:2]
        over, aside = math.hypot(x, y) - POUR_BACK_M, math.atan2(y, x)
        top = max(float(self.container.world_points()[:, 2].max()), self.sand(over, aside))
        carry = max(self.soil.cfg.ground + 0.14, top + CARRY_ABOVE_M if self.pour else 0)
        pace, end = self.pace, start - STROKE_LENGTH_M
        path = [  # (slew, edge distance, edge height, mouth pitch, pace, pause after [s])
            (azimuth, start, far + 0.07, OPEN, pace, 0),
            (azimuth, start, max(far - STROKE_DEPTH_M, bottom), DOWN, 0.7 * pace, 0),
            (azimuth, end, max(near - STROKE_DEPTH_M - 0.01, bottom), math.radians(-150), 0.7 * pace, 0),
            (azimuth, end - 0.03, max(far, near) + 0.08, LEVEL, 0.7 * pace, 0),
            (azimuth, end, carry, LEVEL, pace, 0),
            (aside, over, carry, LEVEL, pace, 0),
            (aside, over, top + POUR_ABOVE_M, DOWN, 0.8 * pace, 0),
            (aside, over, top + POUR_ABOVE_M, EMPTY, 0.8 * pace, self.rng.uniform(0.4, 0.8)),
            (aside, over, carry, LEVEL, pace, 0),
        ]
        if not self.pour:  # lift, and hold it for a second for the load to come to rest
            path = [*path[:4], (*path[4][:5], 1.0)]
        q = np.asarray(joints, float)
        for slew, distance, height, pitch, speed, pause in path:
            q = np.array([slew, *self.arm.solve(distance, height, pitch, q[1:])])
            self.goals.append((q, speed, round(pause * 120)))
        self.strokes.append([round(float(v), 3) for v in (math.degrees(azimuth), start, far, near)])

    def commands(self, targets):
        """Joint -> rate command in -1..1 toward the current goal, all joints arriving together.

        ``targets``: the position targets the commands integrate into (JOINTS order [rad]).
        """
        if not self.goals:
            if self.done:
                return dict.fromkeys(JOINTS, 0.0)
            if self.scoops > 0:  # planned only now: the last scoop changed the sand
                self.scoops -= 1
                self.plan(targets)
            elif self.scoops == 0 and not self.pour:
                self.done = True
                return dict.fromkeys(JOINTS, 0.0)
            elif self.scoops == 0:
                self.scoops = -1
                self.goals.append((self.home, self.pace, 120))  # home, and a second for the sand to settle
            else:
                self.done = True
                return dict.fromkeys(JOINTS, 0.0)
        goal, pace, pause = self.goals[0]
        error = goal - np.asarray(targets, float)
        needed = np.abs(error / self.speed).max() / pace  # seconds at this pace
        if needed < 1 / 120:
            self.goals[0] = (goal, pace, pause - 1)
            if pause <= 0:
                self.goals.pop(0)
            return dict.fromkeys(JOINTS, 0.0)
        # Full pace until 0.15 s short of the goal, then easing in.
        return dict(zip(JOINTS, (error / self.speed / max(needed, 0.15)).tolist()))
