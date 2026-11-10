#!/usr/bin/env bash
# Start Isaac Sim with its bundled ROS 2 Jazzy libs (Kit is Python 3.12; system Lyrical is 3.14 and can't load).
# Do NOT source /opt/ros/lyrical in this shell. Lyrical tools talk to it over DDS (same RMW on both sides).
set -e
ISAACLAB="$(cd "$(dirname "$0")/../../.." && pwd)"
JAZZY="$ISAACLAB/.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib"
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH COLCON_PREFIX_PATH
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$JAZZY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # first, so no other ROS libs win
cd "$ISAACLAB"
exec uv run --extra isaacsim isaacsim "$@"
