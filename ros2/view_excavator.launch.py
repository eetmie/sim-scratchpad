"""RViz for the excavator sim (system ROS 2, e.g. Lyrical).

source /opt/ros/lyrical/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch scripts/sim-scratchpad/ros2/view_excavator.launch.py
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    with open(os.path.join(HERE, "urdf", "excavator.urdf")) as f:
        urdf = f.read()
    sim_time = {"use_sim_time": True}
    return LaunchDescription(
        [
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
