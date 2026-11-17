#!/usr/bin/env bash
# The Newton sim on ROS 2: demo.py --ros with Isaac Sim's bundled ROS 2 Jazzy libs (ros_env.sh; this env is
# Python 3.12, a system Lyrical is 3.14 and can't load). Run from a shell with no /opt/ros sourced.
# Arguments go to demo.py:  sensor_demo/run_ros.sh --view cabin --seed 0
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ros_env.sh"
cd "$HERE/.."
exec uv run --no-sync python sensor_demo/demo.py --ros "$@"
