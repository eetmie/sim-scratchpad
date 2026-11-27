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
from arm_grid import GRID_LINKS, arm_params  # noqa: E402
from robot_boxes import box_params, read_urdf  # noqa: E402

CAMERAS = ["cabin", "arm"]  # FastDEM maps the cabin only (fastdem.yaml); arm points are for inspection
# The robot's arm filter, from kaivuriprokkis native_stack/config/heightmap_pipeline.json (projection section).
ARM_MARGINS_M = (0.090, 0.045, 0.100)  # boom, arm, bucket
ARM_SILHOUETTE_PX = 8  # the robot's 2 strided cells at stride 4
CAMERA_MASK_RADIUS_M = 0.09
ARM_CAMERA_RULES = {"arm.min_view_deg": -5.0, "arm.min_forward_m": 0.24}


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


def depth_to_cloud_params(calibration):
    """depth_to_cloud parameters: self-filter (link boxes + the robot's arm filter) and optional depth correction."""
    urdf = read_urdf(HERE)
    return {
        "cameras": CAMERAS,
        "max_range_m": sensors.DEPTH_MAX_M,
        **box_params(urdf, skip=("lower_carriage", *GRID_LINKS)),
        **arm_params(
            os.path.join(HERE, "..", "assets", "urdf", "excavator.urdf"),
            os.path.join(HERE, "arm_model.npz"),
            ARM_MARGINS_M,
            os.path.join(HERE, "ws", "arm_grids.bin"),
        ),
        "arm_silhouette_px": ARM_SILHOUETTE_PX,
        "camera_mask_radius_m": CAMERA_MASK_RADIUS_M,
        "camera_mask_frames": [sensors.camera_frame(n) for n, *_ in sensors.DEPTH_CAMERAS],
        "base_frame": "upper_carriage",
        **ARM_CAMERA_RULES,
        **depth_correction(calibration),
    }


def depth_to_cloud(context, sim_time):
    params = depth_to_cloud_params(LaunchConfiguration("calibration").perform(context))
    return [
        Node(
            package="excavator_depth_to_cloud",
            executable="depth_to_cloud",
            name="depth_to_cloud",
            parameters=[params, sim_time],
            output="screen",
        )
    ]


def generate_launch_description():
    sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("calibration", default_value="", description="robot sand-calibration.json"),
            OpaqueFunction(function=depth_to_cloud, args=[sim_time]),
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
