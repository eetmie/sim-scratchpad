# Newton soil sensor demo

Newton/MuJoCo-Warp runs the fixed excavator; Warp runs the soil and geometric sensors. No ROS or Kit is needed.

From **sim-scratchpad**, with the existing Isaac Lab uv environment:

```bash
uv run --no-sync python sensor_demo/demo.py --view soil
```

The default sensor is the 64x64 ground height map. The VLA configuration adds the cabin camera, shows it as the
main view instead of the 3D scene, and randomises every episode:

```bash
uv run --no-sync python sensor_demo/demo.py --view cabin --seed 0
```

## Task layout and episodes

The sand lies in a 60 x 60 cm pit, 15 cm deep with a hard bottom (`SoilConfig.floor`), flush with hard ground at the robot's level (`SoilConfig.ground`;
set it equal to `floor` for the old free-standing block). One of 20 SimReady household containers (bowls, mugs,
cups, a jug and a tray: `ASSETS` in `task.py`, with the download command) stands on the ground beside the robot;
without `--seed` it is the first one, a bowl 13 cm across and 6 cm tall that holds about 0.4 L. `task.py` stamps its top surface into the soil's rigid floor, so sand
poured into it stays, levels out and overflows like it would in any open container; `Container.volume(soil)` is
the sand resting in it, printed as `in container` every two seconds.

The bowl is a rigid body (the asset's own 1.14 kg mass and mesh collider): the arm and bucket push it, tip it and
knock it over. The coupling runs one way, machine to bowl to sand:

- The machine's collision meshes are on, but it only hits the bowl. It still passes through the ground, which is
  an invisible plane that carries the bowl (`spawn_fixed_excavator`).
- Whenever the bowl moves more than 1 mm or about half a degree, its top surface as seen from above is stamped
  again (`Soil.restamp`). Sand lying on it is poured level onto the new shape and runs off if that no longer
  holds it; sand it comes to stand on stays buried underneath until it moves away. No sand is created or lost.
- Sand does not push the bowl and adds no weight to it. A bowl lying tipped with its rim on spilled sand scoops
  up more of it than a real one would when it rocks back.
- The joints are stiff and unlimited, so the bucket can press the bowl hard into the ground; it comes back up.
- All containers are in the physics scene from the start. The unused ones hang parked in the air, 20 m away, and
  are put back twice a second: resting on the ground, their contacts more than halved the frame rate.
  `--containers N` spawns only the first N.
- The containers are drawn in one flat colour; their textures are not downloaded.

`--showcase` stands every container beside the pit at once (rows of five, 30 cm apart, two rows on each side),
each in its own colour, and prints which is where. They are all rigid, so the arm can push them; the sand only
follows the first one.

`--seed N` draws a new layout at start and on every reset (B/R, Y/C): which container and its colour (nine named
ones, `COLORS`), written out as the instruction for the dataset, for example
`"task": "dig the sand and move it to the red coffee mug"`; the container 0.45 to 0.65 m from the slew
axis, 50 to 105 degrees to the left (always the same side, `SIDES`: one camera cannot see the pit and the
container together, so the policy has to know which way to slew), clear of the pit, and a sand surface that is never level: a random fill level (0 to 4 cm
below the brim) with four to nine broad mounds and hollows of up to 4 cm all over it, so where to dig has to be
read from the height map. On top of that comes one of three kinds: `rough` (nothing more), `pile` (one heap) or
`dug` (one or two trenches like earlier strokes leave). A hollow deeper than the sand leaves the bottom bare. It is relaxed to rest before the episode starts. Without `--seed`
the layout is the nominal one. Each layout is printed and, with `--record_dir`, appended to `episodes.jsonl`.
The ranges are constants at the top of `task.py`.

## ROS 2

`run_ros.sh` starts the same demo with `--ros`: the sim then plays the robot on the contract's topics
(`sensor_contract.py`, the ones the real machine's `kaivuri_bridge.py` uses), stamped in simulation time.
Arguments go to `demo.py`:

```bash
sensor_demo/run_ros.sh --view cabin --seed 0            # in a shell with no /opt/ros sourced
```

| Topic | Type | Rate | Notes |
| --- | --- | --- | --- |
| `/excavator/command` (in) | `std_msgs/Float32MultiArray` | | `[slew, lift, tilt, scoop]` valve demand in -1..1; stale after 150 ms (wall clock) = stop |
| `/clock`, `/joint_states` | `rosgraph_msgs/Clock`, `sensor_msgs/JointState` | 120 Hz | every physics step; raw joint angles [rad], the state the datasets use |
| `/excavator/cameras/cabin/image_raw` | `sensor_msgs/Image` rgb8 | 30 Hz | frame `cabin_camera`; `arm` too with `--rgb_cameras` |
| `/excavator/heightmap`, `/excavator/heightmap/image` | `sensor_msgs/Image` 32FC1, mono8 | 5 Hz | ground-truth height from the simulated soil surface |
| `/excavator/episode` | `std_msgs/String` JSON | per reset | latched; `{"event": "start", "episode", "time_s", "task", layout...}` and `{"event": "end", "in_container_l", "success", "strokes"}` |

- With `--ros` the joints follow `/excavator/command` only (`ros2/teleop_joy.py`, a policy, ...) and stop inside the
  robot's joint-angle bounds. The window's sticks and keys no longer drive; B/R and Y/C still
  reset. With `--auto_dig` the sim drives itself and publishes its commands on the same topic, so a bag always
  holds the actions.
- Every reset ends the running episode and starts the next one. An episode is a success when at least
  `task.SUCCESS_L` litres rest in the container at its end.
- rclpy comes from Isaac Sim's bundled Jazzy (Python 3.12, no Kit; `ros_env.sh`, which other Python 3.12
  processes such as ros-excavator's policy node can source too); a system Lyrical talks to it over DDS with
  `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` on both sides. At start the sim holds `/clock` at 1 ns for up to 5 s until
  a recorder listens: `ros2 bag record --use-sim-time` subscribes to nothing before it has seen a clock.
- `ros2 launch ros2/view_excavator.launch.py` shows it in RViz. The sim does not publish the normalized
  `/excavator/joint_angles`; that launch's `ros2/joint_angles.py` derives it from `/joint_states` for anything
  that still wants it.
- `run_ros.sh` sets `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`; a recorder needs the same (see the ros-excavator
  README: over the default UDP transport, raw images were lost and delayed).
- Recording and LeRobot datasets: `~/GitHub/ros-excavator` (`record.py`, `bag_to_lerobot.py`, `bag_timing.py`).

### Scripted expert and unattended collection

`--expert` replaces the operator (`expert.py`). Each episode is `--scoops` (3) loops of dig, swing to the
container, pour with the cutting edge 4 cm above its top (or the heap in it), swing back; after the last one the
arm returns to where it started and the sim resets to the next episode. `--episodes N` stops after N.

Where it digs follows from the sand surface alone, which is what a policy has to pick up from the height map:
of all 20 cm strokes that start 0.54 to 0.68 m out within 18 degrees of straight ahead, it takes the one whose
bucket-wide swath lies highest, and cuts 5 cm under that surface, never into the hard bottom. A scoop lowers its
swath, so the next one goes beside it, nearer or farther on the same line, or deeper in the same place if that
is still the highest. Each stroke is planned only when it starts, from the sand as it then lies.

It plans with the URDF's kinematics from what only the sim knows (sand heights, the container's pose and size)
and drives with the same rate commands as a gamepad, at a pace drawn per episode (`--seed`). An episode of three
scoops takes about 140 simulated seconds. Small containers overflow by the second scoop (a scoop is 0.1 to 0.4 L;
the paper cup holds less); `--containers 9` keeps to the bowls.

```bash
uv run --no-sync python sensor_demo/demo.py --view soil --expert --seed 0         # watch it, no ROS
sensor_demo/collect.sh ~/GitHub/ros-excavator/bags/sim-001 --episodes 20 --seed 0  # recorder + sim, headless
```

`collect.sh` starts the bag recorder (system ROS, in its own subshell) and the sim with `--ros --expert
--no_realtime --viz none`, and stops the recorder when the sim is done. Eight three-scoop episodes over all 20
containers all succeeded (0.2 to 1.1 L in the container); recording runs at 1.7x real time and writes 3.6 GB of
bag per episode.

### Digging task (`--task dig`)

A narrower task to show the height map matters: where to dig, and how deep. The pit is nearly bare (0 to 1 cm
of sand over the hard bottom), with one to three heaps 4 to 10 cm high at random places in the expert's reach
(`dig_top` in `task.py`). The instruction is always `"dig where the sand is deepest and lift a full bucket"`. An
episode is one cut, then the load held up level for a second. Then the sim resets. The container still stands
beside the pit, but only as a distractor.

The result is `lifted_l`: the sand at rest more than 3 cm above the brim (`task.lifted`). Success means at least
0.4 L (`DIG_SUCCESS_L`) with the bucket link at least 10 cm over the brim (`BUCKET_UP_M`; the expert holds it at
14 cm), so a bucket resting in a big heap does not count. Measured with seed 0: the expert's cut through the highest swath lifts 0.57 to 0.68 L.
The same cut through the emptiest swath lifts 0.07 to 0.27 L, or 0.49 L when heaps covered the pit. An episode
ends as soon as a full load of grains at rest has been held up, with every arm joint still (under 4 deg/s), for 0.5 s
(`DIG_HOLD_S`), so sand thrown up by a jerk or a load swung past does not count; whoever drives (expert, teleop or a
policy over ROS), or after 25 s as a failure (`DIG_TIMEOUT_S`).

```bash
sensor_demo/collect.sh ~/GitHub/ros-excavator/bags/dig-001 --task dig --episodes 100 --seed 0
sensor_demo/run_ros.sh --view soil --task dig --seed 100   # with policy_node.py, as for the move task
```


## Sensors

`--rgb_cameras` takes camera names (`--rgb_cameras cabin`); with no name it adds both. `--lidars` adds the two
virtual lidars and their occlusion-limited map. Optional geometric depth cameras:

```bash
uv run --no-sync python sensor_demo/demo.py --view soil --depth_cameras
```

RGB-D cameras on the cabin and arm, 640x480 at 30 Hz of simulation time (previewed in the viewer):

```bash
uv run --no-sync python sensor_demo/demo.py --view soil --rgb_cameras
```

Add `--show_lidar` for green ground returns, yellow moving sand, and red robot/support-block returns.
The map/depth panels are movable and collapsible. Previews use small OpenGL textures uploaded once per scan.
Controls remain ISO: left X slew, left Y arm, right X bucket, right Y boom. B/R resets the robot and soil;
Y/C resets soil. Keyboard: A/D, W/S, J/L, I/K respectively.

## What is simulated

- With `--lidars`: two 60x60-degree lidar fans, mounted on cabin and arm using the copied `sensor_contract.py`.
  Default 1-degree spacing, 3,721 rays each, 0.01–0.8 m radial range, 5 Hz simulation time.
- Triangle ray intersections with the same filtered ground and reconstructed moving surface shown in the viewer.
  Visible loose spray uses small octahedral proxies. The robot uses exported URDF visual triangles transformed
  by Newton link poses; the support pedestal also blocks rays. BVHs update for every scan.
- Nearest-hit occlusion happens before robot/support hits are excluded from the soil map. A hidden patch
  stays unknown. Transparent windows are treated as opaque geometry.
- The **virtual lidar map** uses only the latest pair of scans: highest soil return per cell, no history,
  interpolation or hole filling. It is not FastDEM. Sensor positions follow the actual moving links.
- The **ground map** (`oracle` in the recordings) samples the displayed ground over every crop cell: resting
  sand, slow sand lying on it, and the container. The robot, the load in the bucket and falling sand are not in
  it. It is privileged data and does not model sensor visibility. The container rises above the mono8 range
  (`HEIGHTMAP_Z_M` tops out at the cabin origin), so it saturates to white there; the float map is exact.
- Both maps use `upper_carriage` coordinates: x=0.1..0.9 m, y=-0.4..0.4 m, row zero +X, column zero +Y,
  values in local Z metres. NaN means unknown; mono8 uses 0 unknown and 1..255 for -0.30..0 m.
- `--depth_cameras` adds two **160x120 depth-only** pinhole cameras at the scan rate, with calibrated
  field of view from the 640x480 sensor contract. Values are optical Z metres, NaN on a miss, radial
  query limits 0.05..2 m. These use the same geometry and include robot hits. They do not generate RGB
  or stereo noise. The arm mount remains the existing placeholder from the ROS project.

- `--rgb_cameras` adds the **RGB-D cameras** (`rgb.py`): Newton's Warp ray tracer (`newton.sensors.SensorCamera`)
  on a small render-only model, with one directional light, shadows and ambient light. They use the mounts and
  focal lengths in `sensor_contract.py`, scaled by `--rgb_width` (default 640, 4:3), at `--camera_hz` (default 30)
  of simulation time. `CameraRig.rgb_images[name]` is (H, W, 3) uint8 and `depth_images[name]` is optical Z in
  metres with NaN on a miss, with no range limit or noise. The soil is the lidar scene's ground and moving-soil
  meshes, so cameras and lidars see the same surface. The robot is the visual meshes and material colours of
  `excavator_bucket.usd`, not the single-colour URDF export; the lidar still uses the URDF meshes.
  The floor slab and support block are included, so there is a background outside the sandbox.

This is the useful lidar “cheat”: keep geometry, range, view direction and occlusion, omit optical transport,
reflectance, multiple returns, scan motion distortion and sensor noise. The oracle is a separate, stronger cheat.
Surfaces outside the 2.4 m soil grid are currently absent from sensor queries except the robot/support block.
The standard 0.8 m lidar working volume fits the sandbox; the long-range depth views can therefore show missing
background. Updating the excavator asset requires refreshing the corresponding `ros2/urdf` export too.

## Recording and testing

```bash
# 52 simulated seconds: two strokes, return motions, then settling
uv run --no-sync python sensor_demo/demo.py --auto_dig --auto_cycles 2 --max_steps 6240 --depth_cameras --render_hz 30 --record_dir sensor_demo/runs/manual

# Three simulated minutes: seven strokes, without a viewport
uv run --no-sync python sensor_demo/demo.py --viz none --auto_dig --auto_cycles 7 --max_steps 21600 --depth_cameras --no_realtime --record_dir sensor_demo/runs/headless

uv run --no-sync python sensor_demo/inspect_sensors.py sensor_demo/runs/manual
uv run --no-sync --with pytest python -m pytest sensor_demo -q -o markers=kitless
```

With `--rgb_cameras`, `--record_dir` also writes `cameras/<name>/NNNNNN.png` for every camera frame and
`cameras/steps.jsonl` with the frame's simulation time and joint positions [rad].

Runs contain the inherited soil diagnostics and `sensors/metrics.csv`, plus `sensors/scan_*.npz` at each scan.
Each snapshot records time, both float/mono8 maps, world points, origins, hit labels, cabin/body poses and
optional `depth_cabin_m`/`depth_arm_m`. Labels: 0 miss, 1 ground, 2 moving soil, 3 robot/support occluder.
Miss points are finite ray endpoints; consumers must use `labels != 0`. The two lidars occupy consecutive
blocks of equal ray count in the order of `sensor_contract.LIDARS`.

`scan_ms` includes geometry transfer/BVH work, both scans, both maps and optional depth images. It excludes
surface extraction, physics, disk recording and viewport drawing; it is not total frame time. Initial Warp
compilation is much slower than steady state. `inspect_sensors.py` writes `sensor_comparison.png` and
`sensor_metrics.png`; `diagnostics.py` plots the inherited physics telemetry.

Use `--lidar_step_deg 2` for fewer rays, `--lidar_step_deg 0.5` for denser coverage, `--lidar_hz` for scan rate,
and `--render_hz 30` or `--view cabin` to lower viewport cost. Rates are rounded to whole 120 Hz physics steps. The default
simulation has no extra depth-camera work until `--depth_cameras` is supplied. Avoid simultaneous validation
processes when measuring performance.

The source terrain/particle implementation and its tuning are described in [SOIL_MODEL.md](SOIL_MODEL.md).
The sensor demo shares input/render helpers with `../newton` and robot meshes with `../ros2/urdf`; neither
requires importing ROS. The robot asset remains at `assets/excavator_bucket.usd`
relative to sim-scratchpad. [RGB_PLAN.md](RGB_PLAN.md) records how the RGB cameras were built and what is left.

## Speed

RTX 5070 Laptop GPU, one scripted dig cycle plus settling (20 simulated seconds), `--no_realtime`:

| Configuration | Speed |
| --- | ---: |
| Headless, ground map only | 2.4x real time |
| Headless, cabin camera | 1.7x |
| Headless, cabin and arm cameras | 1.5x |

Rigid contacts for the bowl cost 10 to 15% of these. While the bowl is being pushed, re-stamping it adds 3 to 8 ms
per soil tick. Showing the cabin camera (`--view cabin`) instead of the 3D scene was about 25% faster in a window
before the bowl became a rigid body; that has not been measured again.
