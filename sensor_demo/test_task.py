# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The container holds sand, random episodes start at rest, and the ground map shows the container."""

import numpy as np
import pytest
from lidar import ground_map
from sensor_contract import HEIGHTMAP_SIZE, HEIGHTMAP_X_M, HEIGHTMAP_Y_M
from soil import Soil
from task import ASSETS, COLORS, PIT_CLEARANCE_M, SIDES, containers, new_episode

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not containers(1), reason="SimReady assets not downloaded")]


def test_random_episodes_rest_keep_the_container_off_the_pit_and_hold_poured_sand():
    soil, assets = Soil(device="cpu"), containers()
    rng = np.random.default_rng(0)
    sides, used = set(), set()
    for _ in range(12):
        index, layout = new_episode(soil, assets, rng)
        container = assets[index]
        x, y = layout["container_xy_m"]
        sides.add(np.sign(y))
        used.add(layout["container"])
        assert layout["container"] == container.name and layout["colour"] in COLORS
        assert layout["task"] == f"dig the sand and move it to the {layout['colour']} {container.noun}"
        assert np.hypot(max(0.25 - x, 0, x - 0.85), max(abs(y) - 0.30, 0)) > container.radius + PIT_CLEARANCE_M
        assert not soil._relax()  # nothing flows before the robot moves
        assert soil.volume() == pytest.approx(soil.initial_volume)
        assert container.volume(soil) == 0
    assert sides == set(SIDES)
    assert len(used) > 1 or len(assets) == 1
    index, layout = new_episode(soil, assets[:1], rng)  # the bowl: the numbers below are for its size
    container, (x, y) = assets[0], layout["container_xy_m"]
    # Pour 0.1 L over the middle of the container: it must level out inside instead of leaking through the wall.
    inner = soil.footprint & (np.hypot(soil.cx - x, soil.cy - y) < 0.03)
    soil.height[inner] += 1e-4 / (inner.sum() * soil.cfg.cell**2)
    for _ in range(2000):
        if not soil._relax():
            break
    assert container.volume(soil) == pytest.approx(1e-4, rel=1e-3)
    assert soil.height[inner].max() < soil.base[soil.footprint].max()  # below the brim
    # Knocked onto its side it holds nothing: the sand is poured out and runs off, and none is lost.
    total = soil.volume()
    q = np.sqrt(0.5)
    soil.restamp(container.stamp(soil, (x, y, soil.cfg.ground + container.radius, q, 0, 0, q)))
    for _ in range(2000):
        if not soil._relax():
            break
    assert container.volume(soil) < 0.5e-4
    assert soil.volume() == pytest.approx(total, abs=1e-8)
    # Dragged back over where the sand fell, upright: what it covers is buried, not lost and not inside it.
    soil.restamp(container.stamp(soil, (x, y, soil.cfg.ground, 0, 0, 0, 1)))
    assert soil.buried.sum() > 0
    assert soil.volume() == pytest.approx(total, abs=1e-8)


def test_ground_map_shows_the_container_where_the_cabin_sees_it():
    soil, assets = Soil(device="cpu"), containers(1)
    index, layout = new_episode(soil, assets)  # nominal layout: left of the robot, outside the map until it slews
    container = assets[index]
    assert (layout["container"], layout["colour"]) == (ASSETS[0][0], next(iter(COLORS)))
    x, y = np.hypot(*container.pose[:2]), 0.0  # in the frame of a cabin slewed to face it
    slew = np.arctan2(container.pose[1], container.pose[0])
    cabin = np.array([0, 0, 0.1, 0, 0, np.sin(slew / 2), np.cos(slew / 2)], np.float32)
    height = ground_map(soil, soil.render_height(), cabin)
    row, col = np.unravel_index(np.nanargmax(height), height.shape)
    cell = np.ptp(HEIGHTMAP_X_M) / HEIGHTMAP_SIZE
    # Row zero is +X, column zero is +Y; the highest thing on the map is the brim, within its radius of the centre.
    assert abs(HEIGHTMAP_X_M[1] - (row + 0.5) * cell - x) < container.radius + cell
    assert abs(HEIGHTMAP_Y_M[1] - (col + 0.5) * cell - y) < container.radius + cell
    assert np.nanmax(height) == pytest.approx(soil.base[soil.footprint].max() - cabin[2], abs=0.01)
    assert np.nanmedian(height) == pytest.approx(soil.cfg.ground - cabin[2], abs=1e-4)


def test_dig_episodes_keep_the_sand_in_the_zone_at_rest_with_each_kind_of_terrain():
    from task import DIG_INSTRUCTION, DIG_KINDS, DIG_ZONE, lifted

    soil, assets = Soil(device="cpu"), containers(1)
    rng = np.random.default_rng(0)
    kinds, peaks = set(), {}
    outside = soil.block & ~soil.footprint & (np.hypot(soil.cx, soil.cy) > DIG_ZONE[0][1] + 0.10)
    for _ in range(12):
        _, layout = new_episode(soil, assets, rng, dig=True)
        kinds.add(layout["sand"])
        depth = soil.height - soil.cfg.floor
        peaks.setdefault(layout["sand"], []).append(float(depth[soil.block].max()))
        assert layout["task"] == DIG_INSTRUCTION
        assert depth[outside].max() < 0.02  # bare outside the zone, but for what runs off its edge
        assert not soil._relax() and lifted(soil) == 0
    assert kinds == set(DIG_KINDS)
    assert min(peaks["big"]) > max(peaks["dug"]) - 0.02 and min(peaks["big"]) > 0.1  # the big heap stands high


def test_lifted_counts_only_grains_at_rest_over_the_brim():
    from task import LIFT_M, SETTLED_MPS, lifted

    soil = Soil(device="cpu")
    soil.active[:3] = 1
    soil.volumes[:3] = 1e-4
    soil.positions[:3, 2] = soil.cfg.ground + LIFT_M + 0.01, soil.cfg.ground + LIFT_M + 0.01, soil.cfg.ground
    soil.velocities[1] = (0, 0, 10 * SETTLED_MPS)  # thrown up, in flight
    assert lifted(soil) == pytest.approx(1e-4)  # only the first: over the brim and at rest


def test_dig_probe_puts_one_heap_at_the_angle_asked():
    soil, assets = Soil(device="cpu"), containers(1)
    rng = np.random.default_rng(0)
    for angle in (-18.0, 18.0):
        _, layout = new_episode(soil, assets, rng, dig=True, probe=angle)
        depth = np.where(soil.block & ~soil.footprint, soil.height - soil.cfg.floor, 0)
        heap = np.where(depth > 0.02, depth, 0)  # over the thin layer
        centre = np.degrees(np.arctan2((heap * soil.cy).sum(), (heap * soil.cx).sum()))
        assert layout["sand"] == "probe" and not soil._relax()
        # The zone's edge (20 deg) trims the outer side, so it settles to 9-14 deg: at or past training's 9.
        assert 8 < centre * np.sign(angle) < 16
