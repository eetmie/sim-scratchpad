# Soil model and diagnostics

A fixed excavator digs in a 60 x 60 cm sand pit. Newton/MuJoCo-Warp drives the machine and a custom
Warp particle solver handles disturbed soil. These checks run from the [Newton sensor demo](README.md).
Moving soil uses a smooth particle surface by default, shaded with the same rough,
opaque material as the resting ground. `--soil_render grains` keeps the original grain view;
`--soil_render both` overlays grains on the reconstructed surface for inspection.

From **sim-scratchpad**, using the existing Isaac Lab environment:

```bash
uv run --no-sync python sensor_demo/demo.py
```

Or from the Isaac Lab root:

```bash
uv run --no-sync python scripts/sim-scratchpad/sensor_demo/demo.py
```

The shared excavator asset must be at `assets/excavator_bucket.usd`.
Controls and mesh publishing reuse the helpers in `../newton/`.

## Controls

| Input | Action |
| --- | --- |
| Left stick X / A, D | Carriage swing |
| Left stick Y / W, S | Arm |
| Right stick X / J, L | Bucket |
| Right stick Y / I, K | Boom |
| B / R | Reset excavator and soil |
| Y / C | Reset soil only |
| Mouse | Viewport navigation |

The sticks are rate commands; releasing them holds the joint target. Input uses the first
pyglet game controller with a 20% deadzone. Its detected name is printed at startup.
The 8BitDo Ultimate 2C Bluetooth layout has an explicit mapping: its Z/RZ axes
drive the right stick, while GAS/BRAKE remain triggers. Existing SDL mappings take priority.
Keyboard controls work without a controller. WASD viewport flight is disabled to avoid conflicts.
`--view soil` starts closer to the digging area; the default shows the whole excavator.

## Repeatable checks and recording

Run a scripted dig, lift, swing, and dump, then hold still to observe settling. The stroke follows
joint keyframes (`AUTO_DIG` in `demo.py`): the bucket opens, enters teeth-first about 7 cm deep,
drags toward the machine while curling, then swings with the arm extended and dumps beside the block:

```bash
uv run --no-sync python sensor_demo/demo.py --auto_dig --max_steps 1920 --view soil
```

For a longer check, repeat three dig/lift/dump strokes on the same evolving soil,
returning the arm between strokes, then observe 14 seconds of settling (80 seconds total):

```bash
uv run --no-sync python sensor_demo/demo.py --auto_dig --auto_cycles 3 --max_steps 9600 --view soil --record_dir sensor_demo/runs/long
```

Multiple cycles last 22 seconds each, including the return motion. Soil is never reset
between strokes. `--render_hz 30` lowers viewport cost while keeping physics at 120 Hz
and soil at 60 Hz. The final joint targets hold after the requested cycles.

Record a manual session, with upward-moving particles highlighted in red:

```bash
uv run --no-sync python sensor_demo/demo.py --view soil --debug_particles --record_dir sensor_demo/runs/manual
```

The recorder writes:

- `config.json`: soil parameters.
- `metrics.csv`: each soil tick's time, duration, particle count, speeds, contact corrections,
  dynamic volume, volume drift, and capacity limit status.
- `render.csv`: frame duration, surface extraction duration, surface triangle count,
  and physical ground, displayed ground, and particle maximum heights.
- `event_*.npz`: particle positions, velocities, radii, volumes, bucket pose, resting surface,
  and contact corrections when a velocity spike is detected (at most twice per simulated second).
- `frame_*.png`: the actual Newton viewport, once per simulated second. `--capture_every 0`
  disables frames. Headless recording always works without frame capture.
- `render_*.npz`: particle and ground state corresponding to each surface-mode screenshot,
  including the volume fraction assigned to the displayed ground. These help distinguish
  actual particle motion from reconstruction artifacts.

Generate plots after recording:

```bash
uv run --no-sync python sensor_demo/diagnostics.py sensor_demo/runs/manual
```

This writes `summary.png` and, if an event occurred, `worst_event.png` showing the particle motion,
bucket shell and resting surface. Red debug grains indicate upward velocity above 0.5 m/s,
sideways velocity above 0.75 m/s, or unusually fast position changes. Some of these can occur
during normal lifting or falling. The CSV separately counts fast grains away from the bucket,
including sideways motion. `max_raw_speed_m_s` is the attempted speed before the kinetic-energy
limiter; `max_observed_speed_m_s` measures actual displacement of surviving particles between ticks.
Event detection is a diagnostic heuristic, not a physical correctness test.

For a headless cycle and numerical record:

```bash
uv run --no-sync python sensor_demo/demo.py --viz none --auto_dig --max_steps 1920 --no_realtime --record_dir sensor_demo/runs/headless
```

`--screenshot sensor_demo/runs/final.png` saves the final Newton viewport. Captures use host
readback, including on systems without CUDA/OpenGL interoperability. Recording and snapshots
add overhead; compare performance with recording disabled too. The first active particle tick
includes Warp compilation and should be excluded from timing comparisons.
The first surface extraction may also compile kernels. Only the mesh extraction
uses a CUDA graph. Surface reconstruction requires CUDA.

For detailed first-contact screenshots, use a short run and smaller capture interval:

```bash
uv run --no-sync python sensor_demo/demo.py --auto_dig --max_steps 288 --view soil --record_dir sensor_demo/runs/contact --capture_every 0.1
```

## Model and tuning

The resting soil is a 1 cm height grid over a hard floor at z=-0.135 m. The pit extends from
x=0.25 to 0.85 m and y=-0.30 to 0.30 m, with its brim at z=0.015 m (15 cm deep, ~54 L when full).
Uneven fills settle toward a 34-degree angle of repose during reset.

Soil near the bucket becomes 3D particles with explicit bulk volumes. A spatial hash and
position-based contact solver handle particle friction, the ground, and the bucket's open
floor/back/side plates. Bucket poses interpolate across substeps, including carriage swing.
The shell has a 3 mm collision margin, and swept ray checks stop grains crossing a whole
floor or side plate between substeps. Tiny partial shavings stay in the grid until enough
volume is disturbed to form a stable particle; other column layers are evenly partitioned.
Soil in the bucket remains dynamic; there is no separate bucket-load prism or instant dump.
Particles outside the bucket that stay below 0.15 m/s for 0.15 s merge into the height grid when they rest
on their own column or a neighbouring one, so grains wedged against a cut wall do not linger. Transfers
preserve volume, including partial particle layers and particle-pool exhaustion.

External overlap recovery is separated from velocity reconstruction. Granular recovery is
limited by the incoming neighborhood kinetic energy and the moving plate speed, so resolving
initial overlaps cannot power fast grains. Coherent moving soil retains its speed. A 2 m/s
velocity ceiling remains an additional visual stability safeguard, not a calibrated soil law.
GPU substeps replay kernel launches recorded once per tick length; they deliberately do not use a
CUDA graph, because replaying a captured `HashGrid.build` next to the surface extraction changed
results and tripled the tick. The height grid relaxes only inside the sand's bounding window and
stops once the bed rests. Height-grid bookkeeping and particle activation/merging
currently use NumPy, with particle state copied to the host once per soil tick.

Surface mode uses Newton's sparse particle density field and marching cubes. A smooth union
joins that field to the same filtered height grid used for ground rendering; normals blend
into the ground at the join. Duplicate ground triangles are hidden to avoid patch outlines
and depth flicker. Slow particles near the ground contribute a continuous fraction of their
render volume to the shared height surface; the rest remains in the reconstructed mesh.
Carried material stays with the bucket, and fast falling material stays airborne. Ground
contributions cannot raise a column above the envelope of their contributing grains;
compressed grains retain their excess volume in the 3D surface instead of forming a tall
rendered spire. An airborne grain does not increase this ground-height allowance. Isolated
spray uses smaller grains. Density weights use represented bulk volume, rather than collision
radius, and the two surfaces share the same node heights. These are render-only transitions;
extraction does not change particle positions, velocities, or soil volume. Both surfaces use
the same color, roughness and opacity.

| Option | Default | Effect |
| --- | --- | --- |
| `--particles` | 10000 | Maximum simultaneous dynamic particles. At capacity, remaining ground stays intact. |
| `--cell` | 0.01 | Resting grid and nominal particle spacing, in meters. Smaller values increase cost sharply. |
| `--substeps` | 4 | Particle substeps per 60 Hz soil tick. Increase for faster tool movements. |
| `--iterations` | 4 | Contact iterations per substep. Increase for dense loads and less overlap. |
| `--friction` | 0.7 | Grain, bucket and ground friction. |
| `--cohesion` | 0 | Weak attraction from 0 to 1; start with 0 for dry sand. |
| `--bucket_margin` | 0.003 | Bucket shell collision margin in meters. |
| `--soil_render` | surface | `surface`, `grains`, or `both`. |
| `--surface_voxel` | half the cell size | Surface resolution in meters; larger values reduce extraction cost. |
| `--surface_blend` | 1.5 cells | Width of the ground/particle render transition, in meters. |
| `--no_realtime` | off | Remove wall-clock throttling. |

For a higher-cost comparison, try `--substeps 6 --iterations 5`.
The console reports soil tick duration, dynamic particle count, and volume drift every two
simulated seconds. Peak digging cost can exceed the 60 Hz budget on a laptop; this is an initial
visual development model. The grain view uses coarse collision proxies so the dynamics are easy to inspect.

The fixed excavator has no rigid collision response; soil collides with the bucket and resting
ground only. Tracks, boom and carriage do not displace particles. The height field cannot retain
overhangs after material settles. Particle-to-grid transitions, cohesive dirt/clods, finer rendering,
and faster activation/merging are areas for further development.

## Automated physical checks

Tests cover volume conservation at capacity, lift/tip/settle behavior, and the regression where
overlap recovery launches particles near a stationary bucket or cut wall. Fast floor/side-plate
crossings and initial-overlap energy are also checked. They require no display,
Kit, robot asset, or gamepad:

```bash
uv run --no-sync --with pytest python -m pytest sensor_demo/test_soil.py -q -o markers=kitless
SOIL_TEST_DEVICE=cuda:0 uv run --no-sync --with pytest python -m pytest sensor_demo/test_soil.py -q -o markers=kitless
uv run --no-sync --with pytest python -m pytest sensor_demo/test_surface.py -q -o markers=kitless
```

The first command tests CPU stepping; the second runs the same checks on the GPU. Surface
tests use CUDA and check inactive-slot filtering, ground collar normals, unchanged physics,
airborne separation, and compressed-grain spire prevention. Hardware
gamepad input should be checked interactively on the controller you intend to use.
