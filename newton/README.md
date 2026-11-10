# Excavator lidar + gamepad teleop + sand on Newton

Newton port of `../excavator_teleop.py` + `../lidar_demo_script.py`. Newton's MuJoCo-Warp
solver runs the physics, the lidar is a Newton BVH ray cast, and a light sand bed can be dug.
Two front ends:

```bash
# from the Isaac Lab root
uv run --extra isaacsim python scripts/sim-scratchpad/newton/excavator_lidar_teleop.py                  # Kit (RTX)
uv run --extra isaacsim python scripts/sim-scratchpad/newton/excavator_lidar_teleop.py --viz newton_gl   # Newton viewer
```

| | Kit | Newton viewer |
| --- | --- | --- |
| Rendering | RTX, path-traced look | OpenGL, plain shading |
| Speed (with sand) | ~25 fps at 512 x 512 | ~47 fps |
| Gamepad | carb (Kit input) | pyglet controller (SDL mapping) |
| Height map window | omni.ui (`heightmap_view.py`) | ImGui panel (`heightmap_panel.py`) |
| Camera | Kit viewport | mouse (WASD go to the excavator) |

| Option | Effect |
| --- | --- |
| `--viz newton_gl` / `--viz none` | Newton viewer / headless (no window or input; prints map stats) |
| `--viewport_res 512` | Kit: square render size (default 512) |
| `--lidar_step_deg 1` | Ray spacing (default 1 deg = 3721 rays; the RTX demo used 2) |
| `--no_sand` / `--no_rocks` | Leave out the sand bed / the four rocks |
| `--no_realtime` | Do not throttle to wall-clock time |
| `--granular` | Experimental: 2D MPM sand slab under the arm while digging (off by default) |
| `--show_grains` | Draw the 2D slab's grains (debug; normally only the sand mesh is drawn) |
| `--auto_dig` | Scripted dig, swing and dump cycle, repeated every 14 s |
| `--cameras` | Kit: cabin and arm camera feeds in a window (`--camera_size 512 512`, `--camera_hz 15`) |
| `--feeds_only` | With `--cameras`: stop rendering the perspective viewport, so only the two feeds cost GPU time |
| `--record_dir DIR` | With `--cameras`: PNG frames per camera + `steps.jsonl` (time, joint pos/targets, stick commands) |

Controls are the same as the RTX demo (ISO layout, gamepad 0). Left stick moves swing and arm.
Right stick moves bucket and boom. **Y** clears the map; **B** resets the pose, rocks and sand.
Keyboard: A/D swing, W/S arm, I/K boom, J/L bucket, C clear, R reset.

## Sand (`sand.py`)

A deliberately simple model, not granular physics:

- A 1 cm height grid (1.5 x 1.5 m) of sand thickness laid on the terrain. The bed in front of the
  excavator holds ~32 L with two mounds; the rocks are buried in it.
- The bucket is its mid-plane sheet-metal profile (back wall, floor, lip) extruded over its width.
  Sand above the sheet is cut. It fills the bucket (0.57 L) while the opening faces up and there
  is room; otherwise it is pushed aside.
- Tip the opening below horizontal and the load pours out under it.
- Slopes steeper than 34 deg slump. Volume is conserved, except sand poured off the grid.
- The sand has no collision: the bucket moves through it and only the terrain and rocks push back.
- The lidar sees it via a Warp ray march against the grid. The nearer of sand and Newton hit wins.

### Granular slab (`sand2d.py`, experimental, `--granular`)

Physics for the sand while digging, after "Size Doesn't Matter" (Werner et al., 2026): the excavator digs
in its arm plane, so a 2D slab is enough. 0.3 s after the swing stops, the grid strip under the arm
(bucket width) becomes ~5k particles in a 2D explicit MLS-MPM sand model (Drucker-Prager) in Warp, with
the bucket's sheet profile as a moving collider. Every sand step the slab's column volumes are written
back into the grid cells, so the mesh, lidar and height map show the dug surface; the grains are not drawn.
When the swing moves, grains in the bucket become the carried load and the height-grid rules take over.
Cost: ~4 ms per 60 Hz sand step. Full 3D MPM (Newton's implicit MPM coupled to MuJoCo-Warp) was tried and
ran at ~14 ms per 120 Hz step with 31k particles, too slow for real time on this laptop.

## Cameras (`cameras.py`)

Two `CameraCfg` sensors parented to links: `cabin` on `upper_carriage`, `arm` on `tilt_boom` (stick),
with the poses and 10 mm lenses of `cam_cabin` / `cam_boom` in `demo_env.usd`. Kit's RTX renders them (it sees the USD sand mesh; Newton's
Warp camera renderer would only see physics shapes). The lidar debug points are hidden when cameras are on.

## How it maps to the RTX demo

| RTX / PhysX demo | Newton version |
| --- | --- |
| `demo_env.usd`, PhysX | Built from config: excavator asset (free base, resting on the terrain), MuJoCo-Warp |
| `rough_plane` (96 m, 1.3 M tris) | Same mesh and offset, cropped to 3.2 x 2.8 m (`terrain.py`) |
| `OmniLidar`, 60 x 60 deg, 0-0.8 m, 10 Hz | `ShapeIdRaycastSensor`, same mount, fan and range |
| Object-ID self-filter (StableIdMap) | Per-ray shape ID -> body; excavator hits block rays but are not mapped |
| Height map + omni.ui window | Same code: `heightmap.py` (NumPy) + a Kit or ImGui view |

Differences: the base is free and rests on the bumps (demo_env.usd welds it 6 cm above them). The map's zero is the
terrain's nominal level (z = -0.08). Lidar points are green for terrain/sand and orange for self-hits.

## Files

- `excavator_lidar_teleop.py`: scene, physics and main loop.
- `teleop.py`: gamepad and keyboard for both front ends.
- `shape_id_lidar.py`: Newton ray caster plus per-ray shape IDs.
- `sand.py`: sand bed, bucket and the lidar ray march. `sand_render.py` draws both in Kit or the Newton viewer.
- `sand2d.py`: the 2D MPM slab under the arm.
- `cameras.py`: cabin/arm camera configs, feed window and recorder.
- `heightmap.py`: engine-independent remembered height map.
- `heightmap_view.py` (Kit) / `heightmap_panel.py` (Newton viewer): the map window.
- `terrain.py`: downloads `rough_plane` once and caches the cropped patch in `.cache/`.

## Notes

- `NewtonRaycastSensor.data.pos_w` is the carrier body's position without the mount offset, so
  the lidar origin comes from the ray start points instead.
- The sensor's `ray_hits_w` can lag the ray query by one scan. `scan()` uses the query's own
  distance and shape buffers so they always match.
- The Newton viewer does not run pyglet's event loop, so `teleop.py` pumps the controller itself.
  Untested with a real pad.

## Performance (RTX 5070 Laptop)

| Setup | Result |
| --- | --- |
| Kit, full demo before sand (512 x 512) | 1.0x real time, ~25 fps |
| Kit, no lidar, markers or map window | 1.0x real time, ~32 fps |
| Newton viewer, full demo with sand | 1.0x real time, ~47 fps |
| Isaac Sim + PhysX, original demo_env.usd, no scripts | 0.87x, 52 fps |
| Isaac Sim + PhysX + RTX lidar/teleop scripts | 0.60x, 36 fps |

Rendering dominates: one Kit render is ~25 ms, one Newton step ~0.2 ms (~2 ms with Isaac Lab's
per-step Python), and the sand ~100 ms per second. Physics keeps up with the clock and frames are
drawn when time allows. Isaac Sim instead advances a fixed 1/60 s per frame, so it runs slower
than real time below 60 fps.
