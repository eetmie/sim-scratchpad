#!/usr/bin/env bash
# Build the ROS-side workspace ros2/ws: FastDEM (+ grid_map_msgs) at pinned revisions with our fixes
# (fastdem.patch), and our C++ depth_to_cloud (depth_to_cloud_cpp/, linked in).
#   source /opt/ros/<distro>/setup.bash && scripts/sim-scratchpad/ros2/setup_ws.sh
# Works on Jazzy (robot) and Lyrical (laptop). nanoGrid is pinned inside the patch.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
FASTDEM_REV=5cd3e2552dba62be14cd8fe83d5c81261eafd12e   # reviewed in FastDEM_Update_Notes.md
GRID_MAP_REV=d9614992e36c66d3fe67d9125ca105a0a47676fe  # jazzy branch; only grid_map_msgs is built
mkdir -p "$HERE/ws/src" && cd "$HERE/ws/src"
fetch() {  # repo dir rev
    [ -d "$2" ] || git clone -q "$1" "$2"
    git -C "$2" fetch -q origin "$3" 2>/dev/null || true
    git -C "$2" checkout -q "$3"
}
fetch https://github.com/Ikhyeon-Cho/FastDEM.git FastDEM "$FASTDEM_REV"
fetch https://github.com/ANYbotics/grid_map.git grid_map "$GRID_MAP_REV"
git -C FastDEM checkout -q -- . && git -C FastDEM apply "$HERE/fastdem.patch"
ln -sfn ../../depth_to_cloud_cpp depth_to_cloud_cpp
cd "$HERE/ws" && colcon build --packages-up-to fastdem_ros2 excavator_depth_to_cloud --cmake-args -DCMAKE_BUILD_TYPE=Release
