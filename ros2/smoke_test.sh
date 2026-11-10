#!/usr/bin/env bash
# Headless end-to-end check: sim -> bundled Jazzy -> system Lyrical. Prints PASS/FAIL per topic.
#   scripts/sim-scratchpad/ros2/smoke_test.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$(mktemp -t excavator_smoke.XXXX.log)"
JAZZY="$HERE/../../../.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib"
(cd "$HERE/../../.." && env -u AMENT_PREFIX_PATH -u PYTHONPATH ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    LD_LIBRARY_PATH="$JAZZY${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    timeout 400 uv run --extra isaacsim python "$HERE/headless_run.py" 60 >"$LOG" 2>&1) &
SIM=$!
# Kit's own log lines are unbuffered; script prints are not, so wait on the bridge extension line.
until grep -aq "isaacsim.ros2.bridge-.*startup" "$LOG" || ! kill -0 $SIM 2>/dev/null; do sleep 2; done
sleep 20  # graph build, Play, first lidar scans

source /opt/ros/lyrical/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fail=0
for topic in /clock /joint_states /cam_cabin/image_raw /cam_boom/image_raw /lidar_cabin/points /lidar_boom/points \
    /excavator/heightmap /excavator/heightmap_cloud; do
    if timeout 10 ros2 topic echo --once "$topic" >/dev/null 2>&1; then echo "PASS $topic"; else echo "FAIL $topic"; fail=1; fi
done
wait $SIM
grep -aE "\[headless_run\]|AssertionError|ROS_DISTRO=" "$LOG"
echo "log: $LOG"
exit $fail
