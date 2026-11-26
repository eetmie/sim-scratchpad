"""Remote RViz for launch_3dgs.py --ros. The viewer publishes TF and processed maps.

Do not also launch mapping.launch.py or view_excavator.launch.py for this source.
The robot description is published here so mesh paths refer to the RViz machine's checkout.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    sim_time = {"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)}
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                executable=sys.executable,
                arguments=[os.path.join(HERE, "robot_description.py")],
                name="viewer_robot_description",
                parameters=[sim_time],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", os.path.join(HERE, "latest.rviz")],
                parameters=[sim_time],
                output="screen",
            ),
        ]
    )
