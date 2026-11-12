"""Everything ROS-side for the sim: robot_state_publisher, sensor frames, teleop, mapping and RViz.

source /opt/ros/lyrical/setup.bash && source scripts/sim-scratchpad/ros2/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch scripts/sim-scratchpad/ros2/view_excavator.launch.py   # mapping/teleop/rviz:=false to leave out
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sensors  # noqa: E402


def sensor_tf(parent, child, xyz, q):
    """Static TF of a sensor frame. Sim only: the real robot publishes its calibrated camera frames itself."""
    keys = ("--x", "--y", "--z", "--qx", "--qy", "--qz", "--qw", "--frame-id", "--child-frame-id")
    args = [str(a) for pair in zip(keys, (*xyz, *q, parent, child)) for a in pair]
    return Node(package="tf2_ros", executable="static_transform_publisher", name=f"tf_{child}", arguments=args)


def generate_launch_description():
    with open(os.path.join(HERE, "urdf", "excavator.urdf")) as f:
        urdf = f.read()
    sim_time = {"use_sim_time": True}
    frames = [sensor_tf(p, sensors.camera_frame(n), xyz, q) for n, p, xyz, q, *_ in sensors.DEPTH_CAMERAS]
    frames += [sensor_tf(p, n, xyz, q) for n, p, xyz, q in sensors.LIDARS]
    # The robot's bridge publishes these itself; the sim derives them from /joint_states.
    frames.append(Node(executable=os.path.join(HERE, "joint_angles.py"), name="joint_angles"))
    return LaunchDescription(
        frames
        + [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("mapping", default_value="true"),
            DeclareLaunchArgument("teleop", default_value="true"),
            IncludeLaunchDescription(  # gamepad -> /excavator/command, which the sim drives the joints from
                PythonLaunchDescriptionSource(os.path.join(HERE, "teleop.launch.py")),
                condition=IfCondition(LaunchConfiguration("teleop")),
            ),
            IncludeLaunchDescription(  # self-filter + FastDEM, the same launch the robot runs
                PythonLaunchDescriptionSource(os.path.join(HERE, "mapping.launch.py")),
                condition=IfCondition(LaunchConfiguration("mapping")),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": urdf}, sim_time],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(HERE, "excavator.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
                parameters=[sim_time],
            ),
        ]
    )
