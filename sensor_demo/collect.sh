#!/usr/bin/env bash
# Unattended data collection: record a session bag while the scripted expert works through episodes.
#   sensor_demo/collect.sh ~/GitHub/ros-excavator/bags/sim-001 --episodes 20 --seed 0 [more demo.py arguments]
# Headless and as fast as the sim goes by default; add --viz newton_gl to watch. Run from a shell with no ROS
# sourced: the recorder gets the system ROS in its own subshell, the sim Isaac Sim's bundled one (run_ros.sh).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS="${ROS_EXCAVATOR:-$HOME/GitHub/ros-excavator}"
OUT="$1"; shift
SETUP="$(ls /opt/ros/*/setup.bash | head -1)"
( source "$SETUP"; export RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA  # as run_ros.sh
  exec "$HARNESS/record.py" "$HARNESS/profiles/kaivuri.json" "$OUT" --sim > "$OUT.log" 2>&1 ) &
RECORDER=$!
# TERM, not INT: a background job of a script starts with SIGINT ignored, and the recorder leaves it so.
trap 'kill -TERM $RECORDER 2>/dev/null; wait $RECORDER 2>/dev/null' EXIT
"$HERE/run_ros.sh" --viz none --no_realtime --expert "$@"
