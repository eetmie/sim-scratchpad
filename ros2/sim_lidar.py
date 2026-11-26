"""Read the two RTX lidars in demo_env.usd from Kit.

Import ros2.sim_lidar after Kit starts, then call create(stage, "/World/excavator_bucket"). The 60 x 60 deg
scan pattern and 0.8 m range are set on the session layer. new_scan returns Cartesian sensor-frame points
including excavator hits. Call destroy when finished.
"""

import numpy as np

import omni.kit.app
from pxr import Usd, Vt

from .sensors import LIDARS

FAR_M = 0.80
STEP_DEG = 2.0  # ~28 mm between rays at 0.8 m

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.sensors.rtx.nodes", True)
import omni.replicator.core as rep  # noqa: E402
from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data  # noqa: E402
from isaacsim.sensors.experimental.rtx.generic_model_output import (  # noqa: E402
    CoordsType,
    ElementFlags,
    FrameOfReference,
)


def configure(prim):
    angles = np.linspace(-30.0, 30.0, round(60.0 / STEP_DEG) + 1)
    azimuth, elevation = np.meshgrid(angles, angles)
    side, count = len(angles), azimuth.size
    values = {
        "nearRangeM": 0.0,
        "farRangeM": FAR_M,
        "rangeCount": 1,
        "rangesMinM": Vt.FloatArray([0.0]),
        "rangesMaxM": Vt.FloatArray([FAR_M]),
        "minReflectionRangeM": FAR_M,
        "minDistBetweenEchosM": 0.0,
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


def create(stage, robot_path):
    """One render product + GenericModelOutput annotator per lidar in sensors.LIDARS."""
    lidars = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, parent, *_ in LIDARS:
            path = f"{robot_path}/{parent}/{name}"
            prim = stage.GetPrimAtPath(path)
            if not prim or prim.GetTypeName() != "OmniLidar":
                raise RuntimeError(f"{path} must be an OmniLidar in demo_env.usd")
            configure(prim)
            product = rep.create.render_product(
                path, (1, 1), force_new=True, name=f"Lidar_{name}", render_vars=["GenericModelOutput"]
            )
            annotator = rep.annotators.get("GenericModelOutput", device="cpu")
            annotator.attach([product.path])
            lidars.append({"name": name, "product": product, "annotator": annotator, "last": None})
    return lidars


def new_scan(lidar):
    """Sensor-frame points of a scan not returned before, else None."""
    packet = lidar["annotator"].get_data()
    data = packet.get("data") if isinstance(packet, dict) else packet
    if data is None or not data.size:
        return None
    scan = parse_generic_model_output_data(data)
    key = (int(scan.frameId), int(scan.timestampNs))
    if key == lidar["last"]:
        return None
    if scan.elementsCoordsType != CoordsType.CARTESIAN or scan.frameOfReference != FrameOfReference.SENSOR:
        raise ValueError(f"{lidar['name']}: lidar output must be Cartesian in the sensor frame")
    lidar["last"] = key
    points = np.column_stack((scan.x, scan.y, scan.z))
    valid = (
        np.isfinite(points).all(axis=1)
        & ((scan.flags & int(ElementFlags.VALID)) != 0)
        & (np.linalg.norm(points, axis=1) > 0.0)
    )
    return points[valid]


def destroy(lidars):
    for lidar in lidars:
        lidar["annotator"].detach([lidar["product"].path])
        lidar["product"].destroy()
