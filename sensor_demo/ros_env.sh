# Source this to give a Python 3.12 process ROS 2 without a system install of the matching distro: Isaac Sim's
# bundled Jazzy (rclpy and message types). Used by run_ros.sh for the sim; also for the policy node of
# ~/GitHub/ros-excavator, whose LeRobot environment is Python 3.12 while the system ROS (Lyrical) is 3.14.
# Do NOT source /opt/ros/* in the same shell. Other ROS processes talk to it over DDS (same RMW on both sides).
_JAZZY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy"
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# Raw images are 27 MB/s per camera. Over the default UDP transport that lost frames and delayed others by up to
# a second between this Jazzy and a Lyrical recorder; LARGE_DATA moves the data to TCP. Set it on both sides.
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS-LARGE_DATA}"
export LD_LIBRARY_PATH="$_JAZZY/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # first, so no other ROS libs win
export PYTHONPATH="$_JAZZY/rclpy"
