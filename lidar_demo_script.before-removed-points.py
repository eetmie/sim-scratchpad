"""Run in Isaac Sim's Script Editor, then press Play. Rerun to replace this helper's output."""

import asyncio
import time

import carb
import numpy as np
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

LIDAR_NEAR_M = 0.0  # Debugging: accept hits right up to the sensor origin.
LIDAR_FAR_M = 0.80
LIDAR_ANGULAR_STEP_DEG = 2.0  # ~28 mm between rays at 0.8 m (~3 map cells): past ~0.3 m each scan
# leaves gaps between rays, and the remembered map fills them in as the view changes.
# RTX lidar default is 0.4 m, which also acts as a dead zone in front of the sensor:
# hits closer than this still stop the ray (occlusion) but return no point.
LIDAR_MIN_ECHO_SEPARATION_M = 0.0
POINT_SIZE_M = 0.003

# Both lidars feed ONE shared height map: (prim path, debug point color).
SENSORS = [
    ("/World/excavator_bucket/upper_carriage/lidar_cabin", [0.0, 1.0, 0.3, 1.0]),
    ("/World/excavator_bucket/tilt_boom/lidar_boom", [1.0, 0.6, 0.0, 1.0]),
]

# Fixed WORLD-space crop, independent of sensor tilt, carriage swing, or floor prim names.
HEIGHTMAP_SIZE = 64
HEIGHTMAP_X_M = (0.16, 0.80)
HEIGHTMAP_Y_M = (-0.32, 0.32)
GROUND_Z_M = 0.0
HEIGHTMAP_HEIGHT_M = (-0.15, 0.15)  # Fixed brightness scale: ground is middle gray.
HEIGHTMAP_UPDATE_HZ = 5.0
# A remembered cell is cleared when a ray to a farther hit passes this far below its stored height.
CLEAR_MARGIN_M = 0.02

# Self-filter: drop returns whose hit object lives under this prim (sim ground truth via lidar object IDs).
ROBOT_PATH = "/World/excavator_bucket"
# Object IDs are 128-bit; the top 32 bits carry a submesh/primitive index, so match on the low 96 bits.
OBJECT_ID_BASE_MASK = (1 << 96) - 1

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.sensors.rtx.nodes", True)

import omni.replicator.core as rep  # noqa: E402
from isaacsim.sensors.experimental.rtx import (  # noqa: E402
    parse_generic_model_output_data,
    parse_object_ids,
    parse_stable_id_map_data,
)
from isaacsim.sensors.experimental.rtx.generic_model_output import (  # noqa: E402
    CoordsType,
    ElementFlags,
    FrameOfReference,
)


def configure_cabin_lidar(stage, sensor_path):
    """Apply matching global/per-emitter range limits and a denser 60-degree scan."""
    prim = stage.GetPrimAtPath(sensor_path)
    angles = np.linspace(-30.0, 30.0, round(60.0 / LIDAR_ANGULAR_STEP_DEG) + 1)
    azimuth, elevation = np.meshgrid(angles, angles)
    side = len(angles)
    count = azimuth.size
    values = {
        "nearRangeM": LIDAR_NEAR_M,
        "farRangeM": LIDAR_FAR_M,
        "rangeCount": 1,
        "rangesMinM": Vt.FloatArray([LIDAR_NEAR_M]),
        "rangesMaxM": Vt.FloatArray([LIDAR_FAR_M]),
        "minReflectionRangeM": LIDAR_FAR_M,
        "minDistBetweenEchosM": LIDAR_MIN_ECHO_SEPARATION_M,
        "elementsCoordsType": "CARTESIAN",
        "outputFrameOfReference": "SENSOR",
        "instantLidar": True,
        "numberOfEmitters": count,
        "numberOfChannels": count,
        "numLines": side,
        "numRaysPerLine": Vt.UIntArray([side] * side),
        "emitterState:s001:azimuthDeg": Vt.FloatArray(azimuth.ravel()),
        "emitterState:s001:elevationDeg": Vt.FloatArray(elevation.ravel()),
        "emitterState:s001:fireTimeNs": Vt.UIntArray([0] * count),
        "emitterState:s001:channelId": Vt.UIntArray(range(1, count + 1)),
        "emitterState:s001:rangeId": Vt.UIntArray([0] * count),
        "emitterState:s001:bank": Vt.UIntArray(np.repeat(np.arange(side), side).tolist()),
    }
    for name, value in values.items():
        prim.GetAttribute("omni:sensor:Core:" + name).Set(value)
    # EXTRA auxiliary output adds a per-hit object ID (objId), used by the self-filter.
    prim.CreateAttribute("_replicator:rendervar:GenericModelOutput:channels", Sdf.ValueTypeNames.StringArray, True).Set(
        ["EXTRA"]
    )
    prim.SetDocumentation(
        f"Synthetic cabin mapping lidar: 60 x 60 deg FOV, {LIDAR_ANGULAR_STEP_DEG} deg sampling, "
        f"10 Hz snapshot scans, {LIDAR_NEAR_M:.2f}-{LIDAR_FAR_M:.2f} m range. "
        "Not a calibrated hardware model."
    )
    print(
        f"[Cabin lidar] Range {LIDAR_NEAR_M:.2f}-{LIDAR_FAR_M:.2f} m; "
        f"{count} rays/scan; {LIDAR_ANGULAR_STEP_DEG} degree spacing."
    )


def scan_points_in_world(scan, drop):
    """Use the engine's pose stored WITH this snapshot, rather than a newer USD pose.

    GMO quaternions use (x, y, z, w); Gf matrices multiply row vectors.
    Returns a copied point array and the sensor-to-world row-vector transform.
    """
    if scan.elementsCoordsType != CoordsType.CARTESIAN or scan.frameOfReference != FrameOfReference.SENSOR:
        raise ValueError("Height map needs Cartesian, sensor-frame lidar output; rerun the helper.")
    position = np.asarray(scan.frameStart.posM, dtype=np.float64)
    orientation = np.asarray(scan.frameStart.orientation, dtype=np.float64)
    if not np.isfinite(position).all() or not np.isfinite(orientation).all() or np.linalg.norm(orientation) < 1e-6:
        raise ValueError("Waiting for a scan with a valid engine pose.")
    quaternion = Gf.Quatd(float(orientation[3]), Gf.Vec3d(*orientation[:3])).GetNormalized()
    transform = np.asarray(Gf.Matrix4d().SetRotate(quaternion).SetTranslateOnly(Gf.Vec3d(*position)))
    points = np.column_stack((scan.x, scan.y, scan.z))
    ranges = np.linalg.norm(points, axis=1)
    # The sensor already enforces near/far range; only drop invalid or empty elements here.
    valid = np.isfinite(points).all(axis=1) & ((scan.flags & int(ElementFlags.VALID)) != 0) & (ranges > 0.0) & ~drop
    return points[valid] @ transform[:3, :3] + transform[3, :3], transform


def robot_ids_from_map(packet):
    """Base object IDs of every prim under ROBOT_PATH, from the StableIdMap annotator output."""
    data = packet.get("data") if isinstance(packet, dict) else packet
    if data is None or not len(data):
        return set()
    return {
        object_id & OBJECT_ID_BASE_MASK
        for object_id, path in parse_stable_id_map_data(data).items()
        if path == ROBOT_PATH or path.startswith(ROBOT_PATH + "/")
    }


def robot_hit_mask(scan, robot_ids):
    """Per-element mask of returns that hit the excavator itself; exact and lag-free, but sim-only."""
    ids = parse_object_ids(scan.objId)
    if len(ids) != scan.numElements:
        raise ValueError("Scan has no object IDs; rerun the helper (it enables EXTRA lidar output).")
    return np.fromiter(((i & OBJECT_ID_BASE_MASK) in robot_ids for i in ids), dtype=bool, count=len(ids))


def new_heightmap_memory():
    """Empty persistent map: the scan fields plus when each cell was last seen."""
    memory = rasterize_heightmap(np.empty((0, 3)))
    memory["last_seen_ns"] = np.zeros(memory["valid"].shape, dtype=np.int64)
    memory["fresh"] = np.zeros(memory["valid"].shape, dtype=bool)
    memory["stamp_ns"] = 0
    return memory


def merge_heightmap(memory, scan_map, stamp_ns):
    """Cells seen in this scan replace the old value (so digging lowers them); unseen cells keep theirs."""
    seen = scan_map["valid"]
    for key in ("height_m", "gray_u8"):
        memory[key][seen] = scan_map[key][seen]
    memory["valid"] |= seen
    memory["last_seen_ns"][seen] = stamp_ns
    memory["fresh"] = seen
    memory["counts"] = scan_map["counts"]
    memory["stamp_ns"] = stamp_ns
    return memory


def world_to_cells(world):
    """Flat cell index (row-major, row 0 = +X edge, column 0 = +Y edge) for points inside the crop."""
    size = HEIGHTMAP_SIZE
    x0, x1 = HEIGHTMAP_X_M
    y0, y1 = HEIGHTMAP_Y_M
    inside = (
        np.isfinite(world).all(axis=-1)
        & (world[..., 0] >= x0)
        & (world[..., 0] < x1)
        & (world[..., 1] >= y0)
        & (world[..., 1] < y1)
    )
    points = world[inside]
    ix = np.minimum(((points[:, 0] - x0) * size / (x1 - x0)).astype(np.int64), size - 1)
    iy = np.minimum(((points[:, 1] - y0) * size / (y1 - y0)).astype(np.int64), size - 1)
    return (size - 1 - ix) * size + (size - 1 - iy), inside


def clear_seen_through(memory, origin, hits):
    """Forget remembered cells that a lidar ray just passed through, below their stored height.

    A surface there would have blocked the ray, so the stored value is stale (e.g. a bucket hit that
    leaked into the map). Cells hidden behind the bucket keep their value: no terrain ray crosses them.
    Returns how many cells were cleared.
    """
    step = (HEIGHTMAP_X_M[1] - HEIGHTMAP_X_M[0]) / HEIGHTMAP_SIZE
    rays = hits - origin
    reach = np.linalg.norm(rays[:, :2], axis=1)
    # Sample every cell, stopping two cells short of the hit so the hit's own cell is never cleared.
    t = np.arange(1, int(reach.max(initial=0) / step) + 1) * step / np.maximum(reach, 1e-9)[:, None]
    along = t < 1 - 2 * step / np.maximum(reach, 1e-9)[:, None]
    samples = origin + t[along][:, None] * rays[np.nonzero(along)[0]]
    cells, inside = world_to_cells(samples)
    stored = memory["height_m"].ravel()[cells]
    stale = np.unique(cells[stored > samples[inside, 2] - GROUND_Z_M + CLEAR_MARGIN_M])
    for key, empty in (("height_m", np.nan), ("valid", False), ("gray_u8", 0), ("last_seen_ns", 0)):
        memory[key].ravel()[stale] = empty
    return len(stale)


def rasterize_heightmap(world):
    """Highest return per XY cell within ONE scan; merge_heightmap adds the memory across scans.

    Row zero is +world X (far edge); column zero is +world Y (left edge).
    Unobserved cells are NaN in height_m, false in valid, and zero in gray_u8.
    The display adds a checker pattern separately; it is not part of the numeric map.
    """
    size = HEIGHTMAP_SIZE
    cells, inside = world_to_cells(world)
    points = world[inside]
    counts = np.bincount(cells, minlength=size * size).reshape(size, size)
    height = np.full(size * size, -np.inf, dtype=np.float32)
    np.maximum.at(height, cells, points[:, 2] - GROUND_Z_M)
    height = height.reshape(size, size)
    valid = counts > 0
    height[~valid] = np.nan
    gray = np.zeros((size, size), dtype=np.uint8)
    low, high = HEIGHTMAP_HEIGHT_M
    gray[valid] = np.rint(255 * np.clip((height[valid] - low) / (high - low), 0, 1)).astype(np.uint8)
    return {"height_m": height, "valid": valid, "gray_u8": gray, "counts": counts}


class HeightMapView:
    """A UI-only preview: no scene geometry and no additional render product."""

    def __init__(self):
        self.data = new_heightmap_memory()
        self.selected = None
        self.provider = ui.ByteImageProvider()
        self.window = ui.Window("Lidar Height Map", width=460, height=780)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label(f"{HEIGHTMAP_SIZE} x {HEIGHTMAP_SIZE} | world +X up, +Y left", height=22)
                ui.Label(
                    f"X {HEIGHTMAP_X_M[0]:.2f}..{HEIGHTMAP_X_M[1]:.2f} m | "
                    f"Y {HEIGHTMAP_Y_M[0]:.2f}..{HEIGHTMAP_Y_M[1]:.2f} m",
                    height=20,
                )
                ui.Label(
                    f"Cell {(HEIGHTMAP_X_M[1] - HEIGHTMAP_X_M[0]) * 1000 / HEIGHTMAP_SIZE:.1f} x "
                    f"{(HEIGHTMAP_Y_M[1] - HEIGHTMAP_Y_M[0]) * 1000 / HEIGHTMAP_SIZE:.1f} mm | "
                    f"ground Z = {GROUND_Z_M:.3f} m",
                    height=20,
                )
                with ui.HStack(height=384):
                    ui.Spacer(width=ui.Fraction(1))
                    self.image = ui.ImageWithProvider(
                        self.provider,
                        width=384,
                        height=384,
                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                        mouse_pressed_fn=self.inspect_cell,
                    )
                    ui.Spacer(width=ui.Fraction(1))
                ui.Label(
                    f"Black {HEIGHTMAP_HEIGHT_M[0] * 100:.0f} cm | gray 0 | "
                    f"white +{HEIGHTMAP_HEIGHT_M[1] * 100:.0f} cm",
                    height=20,
                )
                ui.Label("Checker = unknown. Blue tint = remembered, not in this scan.", height=20)
                self.status = ui.Label("Press Play to receive scans.", height=36, word_wrap=True)
                self.statistics = ui.Label("No measured cells yet.", height=32, word_wrap=True)
                self.inspection = ui.Label("No cell selected.", height=32, word_wrap=True)
                ui.Label("Remembered map; excavator's own returns removed. Click a cell to inspect.", height=30, word_wrap=True)
        self.update(self.data)

    def update(self, data):
        self.data = data
        yy, xx = np.indices(data["valid"].shape)
        checker = np.where((xx // 2 + yy // 2) % 2, 24, 48).astype(np.uint8)
        display = np.where(data["valid"], data["gray_u8"], checker)
        rgba = np.empty((*display.shape, 4), dtype=np.uint8)
        rgba[:, :, :3] = display[:, :, None]
        rgba[:, :, 3] = 255
        stale = data["valid"] & ~data["fresh"]
        rgba[stale, :2] = (rgba[stale, :2] * 0.7).astype(np.uint8)
        # Enlarge on CPU with nearest-neighbor sampling to keep individual cells legible.
        rgba = np.repeat(np.repeat(rgba, 6, axis=0), 6, axis=1)
        self.provider.set_data_array(rgba.ravel(), [rgba.shape[1], rgba.shape[0]])
        measured = data["height_m"][data["valid"]]
        if measured.size:
            clipped = np.count_nonzero((measured < HEIGHTMAP_HEIGHT_M[0]) | (measured > HEIGHTMAP_HEIGHT_M[1]))
            self.statistics.text = (
                f"Coverage {100 * measured.size / data['valid'].size:.1f}% "
                f"({100 * np.count_nonzero(data['fresh']) / data['valid'].size:.1f}% this scan) | "
                f"height {measured.min() * 1000:+.1f}..{measured.max() * 1000:+.1f} mm | "
                f"{clipped} cells beyond brightness scale"
            )
        else:
            self.statistics.text = "Coverage 0%: no returns inside this world-space crop."
        self.update_inspection()

    def inspect_cell(self, x, y, button, modifiers):
        if button != 0:
            return
        column = int((x - self.image.screen_position_x) * HEIGHTMAP_SIZE / max(self.image.computed_width, 1))
        row = int((y - self.image.screen_position_y) * HEIGHTMAP_SIZE / max(self.image.computed_height, 1))
        if 0 <= row < HEIGHTMAP_SIZE and 0 <= column < HEIGHTMAP_SIZE:
            self.selected = (row, column)
            self.update_inspection()

    def update_inspection(self):
        if self.selected is None:
            return
        row, column = self.selected
        x = HEIGHTMAP_X_M[1] - (row + 0.5) * (HEIGHTMAP_X_M[1] - HEIGHTMAP_X_M[0]) / HEIGHTMAP_SIZE
        y = HEIGHTMAP_Y_M[1] - (column + 0.5) * (HEIGHTMAP_Y_M[1] - HEIGHTMAP_Y_M[0]) / HEIGHTMAP_SIZE
        if not self.data["valid"][row, column]:
            reading = "unknown (never seen)"
        elif self.data["fresh"][row, column]:
            reading = f"{self.data['height_m'][row, column] * 1000:+.1f} mm; {self.data['counts'][row, column]} hits"
        else:
            age_s = (self.data["stamp_ns"] - self.data["last_seen_ns"][row, column]) / 1e9
            reading = f"{self.data['height_m'][row, column] * 1000:+.1f} mm; remembered, seen {age_s:.1f} s ago"
        self.inspection.text = f"[{row}, {column}] X={x:.3f}, Y={y:.3f} m: {reading}"

    def destroy(self):
        self.window.destroy()


def stop_cabin_lidar():
    """Release this helper's render product before constructing another one."""
    state = globals().pop("cabin_lidar_state", None)
    if state is not None:
        # Remove the overlay if an older version is still running in Script Editor.
        if state.get("grid") is not None:
            state["grid"].destroy()
        state["stage_subscription"] = None
        if state.get("task") is not None:
            state["task"].cancel()
        if state.get("view") is not None:
            state["view"].destroy()
        for sensor in state.get("sensors", []):
            for release in (
                sensor["writer"].detach,
                lambda s=sensor: s["annotator"].detach([s["product"].path]),
                lambda s=sensor: s["id_annotator"].detach([s["product"].path]),
                sensor["product"].destroy,
            ):
                try:
                    release()
                except RuntimeError as error:
                    print(f"[Cabin lidar] Resource already closed: {error}")


async def update_cabin_heightmap(state, stage_id):
    """Read fresh snapshots at a bounded rate without retaining any USD prims."""
    app = omni.kit.app.get_app()
    context = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()
    last_update = 0.0
    last_received = time.monotonic()
    last_error = None
    reported = False
    was_playing = False
    while True:
        await app.next_update_async()
        if context.get_stage_id() != stage_id:
            stop_cabin_lidar()
            return
        if not timeline.is_playing():
            if was_playing and timeline.is_stopped():
                # Stop resets the scene, so the remembered terrain is no longer valid.
                state["heightmap"] = new_heightmap_memory()
                state["view"].update(state["heightmap"])
                state["view"].status.text = "Stopped: map cleared."
            elif was_playing:
                state["view"].status.text = "Paused: showing the remembered map."
            was_playing = False
            continue
        if not was_playing:
            for sensor in state["sensors"]:
                sensor["last_scan"] = None
            last_received = time.monotonic()
            state["view"].status.text = "Waiting for a fresh lidar scan..."
            was_playing = True
        now = time.monotonic()
        if now - last_update < 1.0 / HEIGHTMAP_UPDATE_HZ:
            continue
        last_update = now
        if now - last_received > 2.0:
            state["view"].status.text = "No fresh scan for 2 s; showing the remembered map. Check the lidar render product."
        robot_ids = set()
        for sensor in state["sensors"]:
            robot_ids |= robot_ids_from_map(sensor["id_annotator"].get_data())
        robot_ids = robot_ids or state.get("robot_ids", set())
        state["robot_ids"] = robot_ids
        worlds, transforms, keys, removed, error = [], [], [], 0, None
        for sensor in state["sensors"]:
            packet = sensor["annotator"].get_data()
            data = packet.get("data") if isinstance(packet, dict) else packet
            if data is None or not data.size:
                continue
            scan = parse_generic_model_output_data(data)
            key = (int(scan.frameId), int(scan.timestampNs))
            if key == sensor.get("last_scan"):
                continue
            try:
                if not robot_ids:
                    raise ValueError("Waiting for the object-ID map (StableIdMap) to list the excavator.")
                drop = robot_hit_mask(scan, robot_ids)
                world, transform = scan_points_in_world(scan, drop)
            except ValueError as exc:
                error = exc
                continue
            sensor["last_scan"] = key
            worlds.append(world)
            transforms.append(transform)
            keys.append(key)
            removed += int(np.count_nonzero(drop))
        if not worlds:
            if error is not None:
                state["view"].status.text = str(error)
                if str(error) != last_error:
                    print(f"[Cabin lidar] {error}")
                    last_error = str(error)
            continue
        key = max(keys, key=lambda k: k[1])
        last_received = now
        last_error = None
        cleared = sum(
            clear_seen_through(state["heightmap"], t[3, :3], w) for t, w in zip(transforms, worlds)
        )
        world = np.concatenate(worlds)
        heightmap = merge_heightmap(state["heightmap"], rasterize_heightmap(world), key[1])
        heightmap["metadata"] = {
            "frame_id": key[0],
            "timestamp_ns": key[1],
            "sensor_to_world_row_matrices": transforms,
            "world_x_bounds_m": HEIGHTMAP_X_M,
            "world_y_bounds_m": HEIGHTMAP_Y_M,
            "ground_z_m": GROUND_Z_M,
            "grayscale_height_bounds_m": HEIGHTMAP_HEIGHT_M,
            "axes": "row decreases world X; column decreases world Y",
        }
        state["heightmap"] = heightmap
        state["view"].update(heightmap)
        state["view"].status.text = (
            f"Live | frame {key[0]} | {len(world)} terrain returns, {removed} on excavator, {cleared} stale cells cleared | "
            f"max {HEIGHTMAP_UPDATE_HZ:g} updates/s"
        )
        if len(world) and not reported:
            print(f"[Cabin lidar] Height map live: {len(world)} returns; {np.count_nonzero(heightmap['valid'])} cells.")
            reported = True


def start_cabin_lidar():
    """Create one runtime render product and a live, ground-referenced height map."""
    context = omni.usd.get_context()
    stage = context.get_stage()
    for sensor_path, _ in SENSORS:
        if not stage or not stage.GetPrimAtPath(sensor_path):
            raise RuntimeError(f"Open demo_env.usd before running this script; missing {sensor_path}.")
        if stage.GetPrimAtPath(sensor_path).GetTypeName() != "OmniLidar":
            raise RuntimeError(f"Reload the updated demo_env.usd; {sensor_path} must be an OmniLidar.")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise RuntimeError("The height map expects a metre-scaled, Z-up scene.")

    for sensor_path, _ in SENSORS:
        configure_cabin_lidar(stage, sensor_path)

    settings = carb.settings.get_settings()
    print(
        f"[Cabin lidar] Helper v13 (cabin + boom lidars): self-clearing remembered {HEIGHTMAP_SIZE} x {HEIGHTMAP_SIZE} height map, object-ID self-filter; "
        f"motion BVH={settings.get('/renderer/raytracingMotion/enabled')}; "
        f"viewport default tick rate={settings.get('/persistent/app/viewport/defaults/tickRate')}."
    )

    # Session-layer resources are recreated at runtime rather than saved into the scene.
    sensors = []
    state = {"sensors": sensors}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for sensor_path, color in SENSORS:
            name = "LidarDebug_" + sensor_path.rsplit("/", 1)[-1]
            product = rep.create.render_product(
                sensor_path, resolution=(1, 1), force_new=True, name=name, render_vars=["GenericModelOutput"]
            )
            writer = rep.writers.get("RtxSensorDebugDrawPointCloud")
            writer.initialize(color=color, size=POINT_SIZE_M)
            writer.attach([product.path])
            annotator = rep.annotators.get("GenericModelOutput", device="cpu")
            annotator.attach([product.path])
            id_annotator = rep.annotators.get("StableIdMap", device="cpu")
            id_annotator.attach([product.path])
            sensors.append(
                {"product": product, "writer": writer, "annotator": annotator, "id_annotator": id_annotator}
            )

    globals()["cabin_lidar_state"] = state
    try:
        state["view"] = HeightMapView()
        state["heightmap"] = state["view"].data
        state["task"] = asyncio.ensure_future(update_cabin_heightmap(state, context.get_stage_id()))

        def on_stage_event(event):
            if event.type == int(omni.usd.StageEventType.CLOSING):
                stop_cabin_lidar()

        state["stage_subscription"] = context.get_stage_event_stream().create_subscription_to_pop(on_stage_event)
    except Exception:
        stop_cabin_lidar()
        raise
    paths = [x["product"].path for x in sensors]
    print(f"[Cabin lidar] Runtime render products: {paths}. Press Play; green = cabin, orange = boom.")
    print("[Cabin lidar] Open 'Lidar Height Map'; click cells to inspect height. Settings are at the top of the script.")
    print("[Cabin lidar] Numeric map: cabin_lidar_state['heightmap'] (height_m, valid, gray_u8, counts, metadata).")
    return state


stop_cabin_lidar()
cabin_lidar_state = start_cabin_lidar()
