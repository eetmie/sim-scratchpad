"""Height map on the ROS side: depth_to_cloud (C++, self-filter) -> FastDEM. Same launch for the sim and the robot.

source /opt/ros/lyrical/setup.bash && source scripts/sim-scratchpad/ros2/ws/install/setup.bash
ros2 launch scripts/sim-scratchpad/ros2/mapping.launch.py            # sim (use_sim_time:=true)
ros2 launch .../mapping.launch.py use_sim_time:=false calibration:=~/depth-live-bench/sand-calibration.json  # robot
calibration:= loads the stereo depth correction (sand_height.py's depth_correction) into depth_to_cloud; the camera
mounts in that file reach the map through TF (the robot's bridge publishes them), not through here.

Remembered map (world-fixed, follows the undercarriage): /fastdem/mapping/gridmap (grid_map_msgs/GridMap),
/fastdem/mapping/cloud (PointCloud2, for RViz). VLA input (latest scan, turns with the cabin): /excavator/heightmap
(32FC1) and /excavator/heightmap/image (mono8), from heightmap_image.py.
"""

import json
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sensors  # noqa: E402
from robot_boxes import box_params, read_urdf  # noqa: E402

CAMERAS = ["cabin"]


def depth_correction(path):
    """depth_to_cloud parameters for the cameras' depth correction in a robot calibration file ({} for none)."""
    if not path:
        return {}
    with open(os.path.expanduser(path)) as f:
        corrections = json.load(f).get("depth_correction", {})
    params = {}
    for name in CAMERAS:
        fit = corrections.get(sensors.CALIBRATION_KEYS[name])
        if fit:
            params[f"{name}.correction_degree"] = int(fit["degree"])
            params[f"{name}.correction_coeffs"] = [float(c) for c in fit["coeffs"]]
            params[f"{name}.rotated_180"] = name in sensors.ROTATED_180_ON_ROBOT
    return params


def depth_to_cloud(context, urdf, sim_time):
    path = LaunchConfiguration("calibration").perform(context)
    return [
        Node(
            package="excavator_depth_to_cloud",
            executable="depth_to_cloud",
            name="depth_to_cloud",
            parameters=[
                {"cameras": CAMERAS, "max_range_m": sensors.DEPTH_MAX_M, **box_params(urdf), **depth_correction(path)},
                sim_time,
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    urdf = read_urdf(HERE)
    sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("calibration", default_value="", description="robot sand-calibration.json"),
            OpaqueFunction(function=depth_to_cloud, args=[urdf, sim_time]),
            Node(
                executable=os.path.join(HERE, "heightmap_image.py"),
                name="heightmap_image",
                parameters=[sim_time],
                output="screen",
            ),
            Node(
                package="fastdem_ros2",
                executable="fastdem_node",
                name="fastdem",
                parameters=[{"config_file": os.path.join(HERE, "fastdem.yaml")}, sim_time],
                output="screen",
            ),
        ]
    )
