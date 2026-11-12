#!/usr/bin/env bash
# Headless end-to-end check: sim -> bundled Jazzy -> system Lyrical -> FastDEM. Prints PASS/FAIL per topic.
#   scripts/sim-scratchpad/ros2/smoke_test.sh [--lidars]
LIDARS=0; [ "$1" = --lidars ] && LIDARS=1
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$(mktemp -t excavator_smoke.XXXX.log)"
JAZZY="$HERE/../../../.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib"
(cd "$HERE/../../.." && env -u AMENT_PREFIX_PATH -u PYTHONPATH ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    LD_LIBRARY_PATH="$JAZZY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    EXCAVATOR_LIDARS=$LIDARS timeout 400 uv run --extra isaacsim python "$HERE/headless_run.py" 90 >"$LOG" 2>&1) &
SIM=$!
# Kit's own log lines are unbuffered; script prints are not, so wait on the bridge extension line.
until grep -aq "isaacsim.ros2.bridge-.*startup" "$LOG" || ! kill -0 $SIM 2>/dev/null; do sleep 2; done
sleep 20  # graph build, Play, first lidar scans

source /opt/ros/lyrical/setup.bash
source "$HERE/ws/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# TF (robot_state_publisher + sensor frames) and the ROS-side map, as in normal use, without RViz.
ros2 launch "$HERE/view_excavator.launch.py" rviz:=false teleop:=false >"$LOG.mapping" 2>&1 &
sleep 15
fail=0
topics="/clock /joint_states /excavator/joint_angles /excavator/cameras/cabin/image_raw
    /excavator/cameras/arm/depth/image_raw /excavator/cameras/cabin/depth/camera_info /excavator/cameras/cabin/points
    /fastdem/mapping/cloud /excavator/heightmap /excavator/heightmap/image"
[ $LIDARS = 1 ] && topics="$topics /lidar_cabin/points /lidar_boom/points"
for topic in $topics; do
    if timeout 10 ros2 topic echo --once "$topic" >/dev/null 2>&1; then echo "PASS $topic"; else echo "FAIL $topic"; fail=1; fi
done
python3 "$HERE/command_check.py" || fail=1
# P0 from FastDEM_Update_Notes.md: maps must carry the scan time, not 0.
stamp=$(timeout 10 ros2 topic echo --once /fastdem/mapping/cloud --field header.stamp.sec 2>/dev/null | head -1)
if [ -n "$stamp" ] && [ "$stamp" != "0" ]; then echo "PASS map stamp $stamp s"; else echo "FAIL map stamp '$stamp'"; fail=1; fi
wait $SIM
pkill -f "view_excavator.launch.py|mapping.launch.py|teleop.launch.py|ros2/[a-z_]+\.py --ros-args|fastdem_node --ros|depth_to_cloud --ros|robot_state_publisher --ros|static_transform_publisher|game_controller_node" 2>/dev/null
grep -aE "WARN|ERROR|Error" "$LOG.mapping" | sort | uniq -c | head
grep -aE "\[headless_run\]|AssertionError|ROS_DISTRO=" "$LOG"
echo "log: $LOG"
exit $fail
