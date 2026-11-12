"""Gamepad teleop: joy (SDL game controller) -> teleop_joy.py -> /excavator/command. Sim and robot.

ros2 launch scripts/sim-scratchpad/ros2/teleop.launch.py
Record a demo: ros2 bag record /excavator/command /joint_states /tf /tf_static /excavator/cameras/cabin/image_raw ...
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    return LaunchDescription(
        [
            Node(package="joy", executable="game_controller_node", name="joy", parameters=[{"autorepeat_rate": 20.0}]),
            Node(executable=os.path.join(HERE, "teleop_joy.py"), name="teleop_joy", output="screen"),
        ]
    )
