"""Gamepad teleop: joy (SDL game controller) -> teleop_joy.py -> /excavator/command. Sim and robot.

ros2 launch scripts/sim-scratchpad/ros2/teleop.launch.py
Record a demo: ros2 bag record /excavator/command /joint_states /tf /tf_static /excavator/cameras/cabin/image_raw ...
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    joy, command = LaunchConfiguration("joy_topic"), LaunchConfiguration("command_topic")
    return LaunchDescription(
        [
            DeclareLaunchArgument("joy_topic", default_value="joy"),
            DeclareLaunchArgument("command_topic", default_value="/excavator/command"),
            Node(
                package="joy",
                executable="game_controller_node",
                name="joy",
                parameters=[{"autorepeat_rate": 20.0}],
                remappings=[("joy", joy)],
            ),
            Node(
                executable=os.path.join(HERE, "teleop_joy.py"),
                name="teleop_joy",
                output="screen",
                remappings=[("joy", joy), ("/excavator/command", command)],
            ),
        ]
    )
