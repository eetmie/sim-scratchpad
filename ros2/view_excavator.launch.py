"""RViz for the excavator: the Isaac sim or the real robot (kaivuriprokkis native_stack/ros2/kaivuri_bridge.py),
which publish the same topics.

source /opt/ros/<distro>/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch scripts/sim-scratchpad/ros2/view_excavator.launch.py            # sim (sim time)
ros2 launch scripts/sim-scratchpad/ros2/view_excavator.launch.py sim:=false # robot (wall clock)
"""

import os
import re

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    with open(os.path.join(HERE, "urdf", "excavator.urdf")) as f:
        # The exporter writes absolute mesh paths of the machine it ran on; use this checkout's meshes.
        urdf = re.sub(r"file://[^\"]*/urdf/meshes/", f"file://{HERE}/urdf/meshes/", f.read())
    sim_time = {"use_sim_time": ParameterValue(LaunchConfiguration("sim"), value_type=bool)}
    return LaunchDescription(
        [
            DeclareLaunchArgument("sim", default_value="true", description="true: Isaac sim (/clock); false: real robot"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": urdf}, sim_time],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(HERE, "excavator.rviz")],
                parameters=[sim_time],
            ),
        ]
    )
