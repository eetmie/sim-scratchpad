#!/usr/bin/env bash
# Run from a clean shell: bundled Python 3.12 Jazzy for Isaac; external ROS nodes use their own environment.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../sensor_demo/ros_env.sh"
cd "$HERE/.."
exec uv run --no-sync python launch_3dgs.py --ros "$@"
