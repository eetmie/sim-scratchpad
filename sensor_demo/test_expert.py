import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert import JOINTS, Arm, Expert  # noqa: E402

LIMITS = np.radians([(-49.0, 29.0), (29.0, 143.0), (-119.0, 19.0)])  # lift, tilt, tool: the robot's bounds


def test_edge_matches_the_sim_at_the_start_pose():
    # The sim's bucket link sits at [0.641, 0, 0.226] there; its cutting edge is then here.
    assert np.allclose(Arm(LIMITS).edge(*np.radians([-31.5, 71.9, -43.2]))[:2], (0.587, 0.112), atol=1e-3)


def test_arm_solves_back_to_the_edge_it_was_given():
    arm = Arm(LIMITS)
    joints = np.radians([-20.0, 80.0, -40.0])
    edge = arm.edge(*joints)
    assert np.allclose(arm.edge(*arm.solve(*edge, joints + 0.2)), edge, atol=1e-4)


def test_commands_reach_every_goal_and_stop():
    axes = {"revolute_carriage": (0, 0, 0, -15.0), "revolute_lift": (0, 0, 0, 10.0), "revolute_tilt": (0, 0, 0, -12.5),
            "revolute_tool": (0, 0, 0, -20.0)}
    expert = Expert(axes, [(-4, 4), *LIMITS], np.random.default_rng(0))
    home = np.radians([0.0, -31.5, 71.9, -43.2])
    goal = np.radians([60.0, -20.0, 90.0, -5.0])
    expert.goals, expert.done, expert.scoops = [(goal, 1.0, 0), (home, 0.5, 0)], False, -1
    targets, peak, reached = home.copy(), 0.0, False
    for _ in range(120 * 30):
        command = np.array([expert.commands(targets)[name] for name in JOINTS])
        peak = max(peak, np.abs(command).max())
        targets += command * expert.speed / 120  # what the demo does with a command
        reached |= np.allclose(targets, goal, atol=np.radians(0.2))
    assert reached and expert.done and np.allclose(targets, home, atol=np.radians(0.2)) and 0.99 < peak <= 1.0


def test_stroke_goes_where_the_sand_is_and_moves_on_once_it_is_dug():
    from types import SimpleNamespace

    n, cell = 240, 0.01
    y, x = (np.mgrid[:n, :n] + 0.5) * cell - 1.2
    height = np.full((n, n), -0.03) + 0.05 * np.exp(-((x - 0.55) ** 2 + (y - 0.12) ** 2) / (2 * 0.06**2))  # a mound, left
    expert = Expert({name: (0, 0, 0, 10.0) for name in JOINTS}, [(-4, 4), *LIMITS], np.random.default_rng(0))
    expert.soil = SimpleNamespace(height=height, origin=-1.2, cfg=SimpleNamespace(cell=cell, floor=-0.135, ground=0.015))
    azimuth, start, far, near = expert.stroke()
    assert azimuth > np.radians(5) and near > -0.02  # through the mound
    height[np.hypot(x - 0.55, y - 0.12) < 0.12] = -0.09  # dug out
    again = expert.stroke()
    assert abs(again[0] - azimuth) > np.radians(8)  # beside it now


def test_dig_only_stroke_ends_holding_the_load_up():
    from types import SimpleNamespace

    n, cell = 240, 0.01
    y, x = (np.mgrid[:n, :n] + 0.5) * cell - 1.2
    height = np.full((n, n), -0.13) + 0.1 * np.exp(-((x - 0.5) ** 2 + y**2) / (2 * 0.07**2))
    soil = SimpleNamespace(height=height, origin=-1.2, cfg=SimpleNamespace(cell=cell, floor=-0.135, ground=0.015))
    expert = Expert({name: (0, 0, 0, 10.0) for name in JOINTS}, [(-4, 4), *LIMITS], np.random.default_rng(0))
    home = np.radians([0.0, -31.5, 71.9, -43.2])
    bowl = SimpleNamespace(pose=np.array([0.3, 0.5, 0.015]), world_points=lambda: np.array([[0.3, 0.5, 0.1]]))
    expert.start(soil, bowl, home, scoops=1, pour=False)
    targets = home.copy()
    for _ in range(120 * 60):
        if expert.done:
            break
        targets += np.array([expert.commands(targets)[name] for name in JOINTS]) * expert.speed / 120
    assert expert.done and len(expert.strokes) == 1
    distance, height_m, pitch = expert.arm.edge(*targets[1:])
    assert height_m >= 0.015 + 0.14 - 0.01 and abs(np.degrees(pitch) + 176) < 3  # up, bucket level, not home
