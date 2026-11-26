"""Robot state publisher, teleop and RViz for the Newton demo or the real robot.

source /opt/ros/<distro>/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch scripts/sim-scratchpad/ros2/view_excavator.launch.py              # sim (sim time)
ros2 launch scripts/sim-scratchpad/ros2/view_excavator.launch.py sim:=false   # robot (wall clock)
teleop/rviz:=false leave those out. sim:=false also leaves out the sim-only sensor frames and joint_angles:
the robot's bridge publishes its calibrated frames and its own /excavator/joint_angles.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sensors  # noqa: E402
from robot_boxes import read_urdf  # noqa: E402

SIM = IfCondition(LaunchConfiguration("sim"))


def sensor_tf(parent, child, xyz, q):
    """Static TF of a sensor frame. Sim only: the real robot publishes its calibrated camera frames itself."""
    keys = ("--x", "--y", "--z", "--qx", "--qy", "--qz", "--qw", "--frame-id", "--child-frame-id")
    args = [str(a) for pair in zip(keys, (*xyz, *q, parent, child)) for a in pair]
    return Node(
        package="tf2_ros", executable="static_transform_publisher", name=f"tf_{child}", arguments=args, condition=SIM
    )


def generate_launch_description():
    urdf = read_urdf(HERE)
    sim_time = {"use_sim_time": ParameterValue(LaunchConfiguration("sim"), value_type=bool)}
    frames = [sensor_tf(p, sensors.camera_frame(n), xyz, q) for n, p, xyz, q, *_ in sensors.DEPTH_CAMERAS]
    frames += [sensor_tf(p, n, xyz, q) for n, p, xyz, q in sensors.LIDARS]
    # The robot's bridge publishes these itself; the sim derives them from /joint_states.
    frames.append(Node(executable=os.path.join(HERE, "joint_angles.py"), name="joint_angles", condition=SIM))
    return LaunchDescription(
        [DeclareLaunchArgument("sim", default_value="true", description="true: Newton sim (/clock); false: robot")]
        + frames
        + [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("teleop", default_value="true"),
            IncludeLaunchDescription(  # gamepad -> /excavator/command, which the sim drives the joints from
                PythonLaunchDescriptionSource(os.path.join(HERE, "teleop.launch.py")),
                condition=IfCondition(LaunchConfiguration("teleop")),
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
