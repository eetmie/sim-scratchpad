# Latest calibrated viewer on ROS 2

`launch_3dgs.py --ros` publishes the current two-camera robot mapper and FastDEM results.
Depth rendering, robot masks, cabin-first fusion and FastDEM stay on the simulator machine.
Raw depth and grayscale cameras are opt-in; remote RViz needs only maps and TF.

The transport uses `rclpy`, `tf2_ros` and standard ROS messages. `viewer_bridge.py` has no
Isaac/USD imports; `viewer_scene.py` reads USD and optional calibrated grayscale buffers;
`map_contract.py` contains ROS-free array/time conversions. No custom interfaces or ROS
workspace build is required for this path. Isaac's bridge supplies its ROS runtime.

## Ubuntu setup and launch

Use this checkout with its LFS assets and Isaac Lab's existing Python 3.12 / Isaac Sim 6.2
environment. Clone the matching `kaivuriprokkis` checkout as well. Camera/filter
settings and the small motion demo are included; no dataset copy is required. The existing
`ros2/setup_ws.sh` is for the separate ROS-side depth mapper; it is unnecessary here.

Build the ROS-free FastDEM adapter on Ubuntu (the Windows executable cannot be reused):

```bash
export KAIVURI_REPO="$HOME/Documents/GitHub/kaivuriprokkis"  # adapt to the Ubuntu checkout
uv run --no-sync --with cmake --with ninja python build_fastdem.py --robot-repo "$KAIVURI_REPO"
```

In a **clean simulator terminal**, without `/opt/ros/*` sourced:

```bash
export ROS_DOMAIN_ID=42
bash ros2/run_latest.sh --mode replay --robot-repo "$KAIVURI_REPO"
```

`run_latest.sh` selects Isaac's bundled Jazzy and Fast DDS using the existing `ros_env.sh`.
`--ros` implies `--depth-maps` and requires the robot backend. Its default Linux adapter
path is `.cache/fastdem/build/fastdem_replay`, with a fallback to the existing
`logs/fastdem/build/fastdem_replay`; override with `--fastdem PATH`.
The launcher enables `isaacsim.ros2.bridge` before importing `rclpy`.

`--mode replay` loops the bundled pose sequence using its relative timestamps; omit it
for the stationary scene. Both use `config/heightmaps/` by default. `--config PATH`
selects another profile. To use recorded dataset calibration and a saved plan, add
`--dataset PATH` and
`--replay logs/depth_replay/episode-002/replay.json`. Plans live in ignored `logs/`,
so copy yours to Ubuntu or generate a new one from the dataset:

```bash
mkdir -p logs/ros-replay
export KAIVURI_DATASET="$HOME/Desktop/heightmap_tests_complete"
uv run --no-sync python -m camera_calibration.recorded_poses "$KAIVURI_DATASET" \
  --episode 2 --start 12 --stop 672 --stride 6 --timestamps depth --out logs/ros-replay/episode-002.json
bash ros2/run_latest.sh --dataset "$KAIVURI_DATASET" --robot-repo "$KAIVURI_REPO" \
  --replay logs/ros-replay/episode-002.json --ros-cameras
```

On the RViz machine, keep another checkout including `assets/urdf/meshes`. Source that
machine's ROS environment, then run from the scratchpad checkout:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
ros2 launch ./ros2/view_latest.launch.py
```

Install your distro's `rviz2`, `tf2_ros`, ROS Python messages, and Python NumPy packages if
missing. Network discovery, firewall and matching DDS settings must be checked on Ubuntu.
The description publisher runs beside RViz so mesh URIs refer to that machine's checkout.

**Do not also run `view_excavator.launch.py`, `mapping.launch.py`, `heightmap_image.py`,
or another `/clock` publisher for this source.** The viewer owns the complete TF tree and
map outputs. The new launch does not start robot_state_publisher, the older camera TF,
the older cabin-only estimator, or teleop.

## Topics and numeric conventions

**Real-robot matching checklist:** keep these names and contracts identical on both sources:

* Observations: `/excavator/heightmap` + `/image`, `/fastdem/mapping/{height,image,age}`,
  `/excavator/maps/metadata`, and optionally `/excavator/cameras/{cabin,arm}/{image_raw,camera_info}`.
* Robot/RViz: `/joint_states`, `/tf` (plus `/tf_static` when used), and `/fastdem/mapping/cloud`.
* Future policy/control: `/excavator/joint_angles` (`Float32MultiArray`, normalized
  `[slew,lift,tilt,scoop]`), `/excavator/feedback_healthy` (`Bool`), and
  `/excavator/command` (`Float32MultiArray`, same order, demands -1..1). The latest viewer
  currently publishes `/joint_states`; normalized feedback and command handling still need wiring.

Match message types, optical/map frames, ground-relative heights, image encodings, QoS and
acquisition-time stamps too. Then `ROS_DOMAIN_ID` can select the source. Consumers use
`use_sim_time:=true` for the simulator and `false` for the real robot's system-time stamps
(`view_latest.launch.py use_sim_time:=false`); this clock setting must switch as well.

| Topic | Standard message / meaning |
| --- | --- |
| `/clock` | `rosgraph_msgs/Clock`, one authoritative data timeline |
| `/tf` | Actual USD body poses, optical frames and ground-relative map frames |
| `/joint_states` | `sensor_msgs/JointState`, URDF angles from current USD poses |
| `/excavator/heightmap` | `sensor_msgs/Image`, 64x64 `32FC1` metres, current front observation |
| `/excavator/heightmap/image` | `Image`, `mono8` front preview |
| `/fastdem/mapping/height` | `Image`, 160x160 `32FC1` metres, accumulated FastDEM surround |
| `/fastdem/mapping/image` | `Image`, `mono8` surrounding preview |
| `/fastdem/mapping/age` | `Image`, `16UC1`, accepted paired updates; 65535 unknown |
| `/fastdem/mapping/cloud` | `sensor_msgs/PointCloud2`, known surround cell centres, XYZ metres |
| `/excavator/maps/metadata` | Transient-local `std_msgs/String` JSON: bounds, resolution, frames, units, clock mode |
| `/excavator/time/host_capture` | `sensor_msgs/TimeReference`: data stamp to host Unix UTC acquisition time |
| `/excavator/time/host_monotonic` | `TimeReference`: data stamp to host steady acquisition time |
| `/excavator/time/source_capture` | Replay only: original selected exposure's host-monotonic time |
| `/excavator/episode` | Transient-local `std_msgs/String` JSON start/end; one episode per replay loop |

Float heights use NaN for unknown cells. Rows decrease X, columns decrease Y. Geometry
comes from the dataset and is published in metadata. Ground is zero height; previews use
ground=128, -120 mm=16, +120 mm=240, unknown=0 (or the configured colour limit).
`excavator_front_map` is level at the ground reference with cabin yaw;
`excavator_surround_map` is level with base yaw. Their TF places maps into stage `world`.
These differ from the older `upper_carriage`-relative height convention.

Sensor/map publishers use reliable, keep-last depth 1. Metadata and events are
transient-local. RViz requests best effort and depth 1, compatible with these publishers.
The cloud defaults to 2 Hz in data time and is built only when subscribed; use
`--ros-cloud-hz 0` for image-only operation. There is no `grid_map_msgs/GridMap` publisher:
numeric height/age images plus metadata expose the grid without custom-message builds
inside Isaac. Standard RViz uses the PointCloud2 representation.

## Multiple time references, one ROS clock

`--ros-time auto` is the default:

* With `--replay PLAN`, visit **every saved cabin replay pose in order**, without wall-time
  frame skipping. Headers preserve original exposure intervals. Slow rendering therefore
  does not slow recorded motion in a converted dataset. Both synthetic cameras observe
  the same applied pose; the source reference identifies that selected replay exposure,
  not two independently replayed real camera exposures.
* Without replay, advance by `1 / map_hz` per captured observation. For live manual scene
  movement choose `--ros-time wall` to retain actual elapsed host acquisition intervals.

`--ros-time fixed` explicitly selects fixed/replay stepping. `wall` retains wall-paced
replay, which can skip saved poses when rendering is slow. Headers, TF and `/clock` use
one timeline starting at 1 second. Replay loops clear FastDEM and begin new episodes,
while ROS time continues increasing. UTC, host steady time and original source time
remain separate TimeReference values. Never alternate them on `/clock`.

Two source frames 0.2 seconds apart remain 0.2 seconds apart in their headers even if
they take 0.6 seconds each to render. For LeRobot v3 conversion, group by episode,
subtract its first data stamp, align modalities by acquisition headers and resample to
the chosen FPS. Do not assign 5 Hz frames consecutive 30 FPS indices: that speeds motion
up sixfold. Preserve source/host references as extra features or a sidecar when needed.
Video timing and tabular timestamps must agree with the dataset FPS; 30 Hz resampling
does not create additional independent camera observations.

See the [LeRobot v3 format](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3)
and [dataset API](https://huggingface.co/docs/lerobot/en/api/datasets).

Start recording before the simulator. Discovery gets a 3-second grace period, adjustable
with `--ros-start-delay SECONDS`:

```bash
ros2 bag record --use-sim-time -o logs/rosbags/latest \
  /clock /tf /joint_states /excavator/episode /excavator/maps/metadata \
  /excavator/time/host_capture /excavator/time/host_monotonic /excavator/time/source_capture \
  /excavator/heightmap /excavator/heightmap/image \
  /fastdem/mapping/height /fastdem/mapping/image /fastdem/mapping/age
```

Use matching domain/middleware settings. Headers remain the acquisition-time authority
even if bag receipt times differ due to scheduling. During playback stop the simulator
and use one clock owner. If `ros2 bag play --clock` generates `/clock`, exclude recorded
`/clock` from the selected playback topics.

## Optional sensor streams

`--ros-cameras` publishes `/excavator/cameras/<name>/{image_raw,camera_info}`. Images are
calibrated `mono8` at map capture cadence, not an independent 30 Hz. The arm retains
recorder orientation and fitted appearance; its optical TF, principal point and
tangential coefficients match that rotation. IR2 and depth have separate frames,
preserving the arm stereo baseline. Add camera topics explicitly when recording them.

`--ros-depth` publishes `/excavator/cameras/<name>/depth/{image_raw,camera_info}`.
Depth is converted from recorder 0.1 mm Z16 into ROS `16UC1` millimetres and unrotated
to match the physical `<name>_camera` optical frame. Zero is invalid. Ideal rendered
depth has no empirical real-camera correction. The default RViz profile has no depth
or raw point-cloud subscriptions; camera displays start disabled.

## Validation and remaining Ubuntu work

CPU contract tests require no ROS installation or Kit renderer:

```bash
uv run --no-sync python -m pytest -q --noconftest tests/test_ros_viewer.py tests/test_depth_mapping.py
```

On Ubuntu, verify bridge startup and then:

```bash
ros2 topic info /fastdem/mapping/cloud --verbose
ros2 topic echo /excavator/maps/metadata --once --qos-durability transient_local
ros2 run tf2_ros tf2_echo world excavator_surround_map --ros-args -p use_sim_time:=true
ros2 bag info logs/rosbags/latest
```

Compare numeric height/age grids with `latest.npz` at the same sample; verify world
registration in RViz and increasing acquisition stamps across replay loops. Measure
wall throughput separately from simulation-time topic rates. Depth rendering and
local mapping still consume GPU/CPU time.

The adapter reads **paused USD and kinematic replay**. Playing the physics timeline
is rejected because live PhysX/Fabric pose synchronization has not been integrated.
ROS valve commands, action recording and dynamics remain integration work.
Live ROS/DDS/RViz behavior still needs validation on the Ubuntu machine.
