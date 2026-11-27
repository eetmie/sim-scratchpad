"""Inspect the demo's cameras, import native IR metadata, and author a calibration layer."""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from ros2.sensors import CALIBRATION_KEYS, CAMERA_PRIMS, DEPTH_CAMERAS, DEPTH_SIZE
from scipy.spatial.transform import Rotation

from pxr import Gf, Sdf, Usd, UsdGeom

from . import BASE_STAGE

ROBOT = "/World/excavator_bucket"
LENS = "omni:lensdistortion:opencvPinhole"
BROWN = ("k1", "k2", "p1", "p2", "k3")
HIGHER_ORDER = ("k4", "k5", "k6", "s1", "s2", "s3", "s4")
OPTICAL_TO_USD = np.diag([1.0, -1.0, -1.0])
IMAGE_ROTATION = "irCalibration:imageRotationDeg"
LUMA_RESPONSE = "irCalibration:lumaResponse"
MOUNT_OP = "xformOp:transform:irCalibration"


def cameras(stage: Usd.Stage) -> dict[str, UsdGeom.Camera]:
    rig = {}
    for name, link, *_ in DEPTH_CAMERAS:
        path = f"{ROBOT}/{link}/{CAMERA_PRIMS[name]}"
        camera = UsdGeom.Camera.Get(stage, path)
        if not camera:
            raise ValueError(f"Expected a USD camera at {path}")
        rig[name] = camera
    return rig


def pose(matrix: Gf.Matrix4d, meters_per_unit: float) -> dict:
    # Gf uses row vectors. The saved quaternion maps right/down/forward optical axes into the parent.
    transform = np.asarray(matrix).T
    rotation = transform[:3, :3]
    # Replay matrices are rounded to six decimals; composition carries that error into camera poses.
    # Accept only roundoff here; Rotation.from_matrix projects it onto a proper rotation below.
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise ValueError("Camera pose must be rigid")
    return {
        "translation_m": (transform[:3, 3] * meters_per_unit).tolist(),
        "quaternion_xyzw": Rotation.from_matrix(rotation @ OPTICAL_TO_USD).as_quat().tolist(),
    }


def describe(camera: UsdGeom.Camera) -> dict:
    prim = camera.GetPrim()
    units = UsdGeom.GetStageMetersPerUnit(prim.GetStage())
    model = prim.GetAttribute("omni:lensdistortion:model").Get()
    if model == "opencvPinhole":
        if any(prim.GetAttribute(f"{LENS}:{key}").Get() for key in HIGHER_ORDER):
            raise ValueError(f"{prim.GetPath()}: expected five Brown–Conrady coefficients")
        width, height = prim.GetAttribute(f"{LENS}:imageSize").Get()
        intr = {key: prim.GetAttribute(f"{LENS}:{key}").Get() for key in ("fx", "fy", "cx", "cy")}
        coeffs = [prim.GetAttribute(f"{LENS}:{key}").Get() for key in BROWN]
    elif not model and camera.GetProjectionAttr().Get() == "perspective":
        width, height = DEPTH_SIZE
        lens = camera.GetCamera()
        intr = {
            "fx": lens.focalLength * width / lens.horizontalAperture,
            "fy": lens.focalLength * height / lens.verticalAperture,
            "cx": (width - 1) / 2 - lens.horizontalApertureOffset * width / lens.horizontalAperture,
            "cy": (height - 1) / 2 + lens.verticalApertureOffset * height / lens.verticalAperture,
        }
        coeffs = [0.0] * 5
    else:
        raise ValueError(f"{prim.GetPath()}: unsupported camera model {model!r}")
    data = {
        "prim": str(prim.GetPath()),
        "parent": str(prim.GetParent().GetPath()),
        "mount": pose(camera.GetLocalTransformation(), units),
        "optical_to_world": pose(camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), units),
        "image_rotation_deg": prim.GetAttribute(IMAGE_ROTATION).Get() or 0,
        "intrinsics": {
            "width": width,
            "height": height,
            **intr,
            "model": "brown_conrady" if any(coeffs) else "none",
            "coeffs": coeffs,
        },
    }
    if response := prim.GetAttribute(LUMA_RESPONSE).Get():
        data["luma_response"] = list(response)
    return data


def intrinsics(data: dict) -> tuple[np.ndarray, np.ndarray]:
    if any(type(data[key]) is not int or data[key] <= 0 for key in ("width", "height")):
        raise ValueError("Image dimensions must be positive integers")
    fx, fy, cx, cy = (float(data[key]) for key in ("fx", "fy", "cx", "cy"))
    if not all(map(math.isfinite, (fx, fy, cx, cy))) or min(fx, fy) <= 0:
        raise ValueError("Intrinsics must be finite and focal lengths positive")
    coeffs = np.asarray(data["coeffs"], dtype=float)
    if coeffs.shape != (5,) or not np.isfinite(coeffs).all():
        raise ValueError("Expected five finite coefficients in k1,k2,p1,p2,k3 order")
    if data["model"] not in ("none", "brown_conrady") or (data["model"] == "none" and np.any(coeffs)):
        raise ValueError(f"Invalid or unsupported distortion model {data['model']!r}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), coeffs


def native_ir(session: Path, key: str) -> dict:
    info = json.loads((session / f"{key}.depth.json").read_text())
    ir = info["ir_intrinsics"]
    lens = {field: ir[field] for field in ("width", "height", "fx", "fy", "coeffs")}
    model = ir["model"].lower().replace(" ", "_")
    lens.update(cx=ir["ppx"], cy=ir["ppy"], model=model)
    if model in ("brown_conrady", "modified_brown_conrady", "inverse_brown_conrady") and not any(lens["coeffs"]):
        lens["model"] = "none"
    intrinsics(lens)
    return {
        "intrinsics": lens,
        "image_rotation_deg": info["rotation_deg"],
        "native_stream": {
            "key": key,
            **{field: info[field] for field in ("serial", "ir_index", "depth_to_ir_extrinsics") if field in info},
        },
    }


def inspect(stage: Usd.Stage, session: Path | None) -> dict:
    rig = {name: describe(camera) for name, camera in cameras(stage).items()}
    if session is not None:
        for name, camera in rig.items():
            camera.update(native_ir(session, CALIBRATION_KEYS[name]))
    return {
        "stage": stage.GetRootLayer().realPath,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "layers": sorted(layer.realPath for layer in stage.GetUsedLayers() if not layer.anonymous),
        "gaussian_splats": [
            str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "ParticleField3DGaussianSplat"
        ],
        "cameras": rig,
    }


def author(camera: UsdGeom.Camera, data: dict) -> None:
    prim = camera.GetPrim()
    if data["prim"] != str(prim.GetPath()) or data["parent"] != str(prim.GetParent().GetPath()):
        raise ValueError(f"Calibration targets the wrong prim or parent for {prim.GetPath()}")
    lens = data["intrinsics"]
    matrix, coeffs = intrinsics(lens)
    width, height = lens["width"], lens["height"]
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    image_rotation = data["image_rotation_deg"]
    if type(image_rotation) is not int or image_rotation not in (0, 180):
        raise ValueError("image_rotation_deg must be 0 or 180")

    mount = data["mount"]
    xyz = np.asarray(mount["translation_m"], dtype=float)
    q = np.asarray(mount["quaternion_xyzw"], dtype=float)
    if xyz.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[xyz, q]).all():
        raise ValueError("Invalid optical-to-parent mount")
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6):
        raise ValueError("Expected a unit quaternion in xyzw order")
    usd_rotation = Rotation.from_quat(q).as_matrix() @ OPTICAL_TO_USD
    units = UsdGeom.GetStageMetersPerUnit(prim.GetStage())
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(Gf.Matrix3d(*usd_rotation.T.ravel()))
    transform.SetTranslateOnly(Gf.Vec3d(*(xyz / units)))
    if attr := prim.GetAttribute(MOUNT_OP):
        op = UsdGeom.XformOp(attr)
    else:
        op = camera.AddTransformOp(opSuffix="irCalibration")
    op.Set(transform)
    camera.SetXformOpOrder([op])

    focal = camera.GetFocalLengthAttr().Get()
    horizontal, vertical = focal * width / fx, focal * height / fy
    camera.GetHorizontalApertureAttr().Set(horizontal)
    camera.GetVerticalApertureAttr().Set(vertical)
    camera.GetHorizontalApertureOffsetAttr().Set(((width - 1) / 2 - cx) * horizontal / width)
    camera.GetVerticalApertureOffsetAttr().Set((cy - (height - 1) / 2) * vertical / height)
    prim.CreateAttribute(IMAGE_ROTATION, Sdf.ValueTypeNames.Int, custom=True).Set(image_rotation)
    if "luma_response" in data:
        luma = np.asarray(data["luma_response"], dtype=float)
        if luma.shape != (6,) or not np.isfinite(luma).all() or luma[2] <= 0:
            raise ValueError("Expected six finite IR luma coefficients with a positive gamma")
        prim.CreateAttribute(LUMA_RESPONSE, Sdf.ValueTypeNames.FloatArray, custom=True).Set(luma.tolist())
    prim.AddAppliedSchema("OmniLensDistortionOpenCvPinholeAPI")
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    prim.CreateAttribute(f"{LENS}:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))
    values = dict.fromkeys(HIGHER_ORDER, 0.0)
    values.update(fx=fx, fy=fy, cx=cx, cy=cy)
    values.update(zip(BROWN, coeffs, strict=True))
    for key, value in values.items():
        prim.CreateAttribute(f"{LENS}:{key}", Sdf.ValueTypeNames.Float).Set(float(value))


def apply(base: Path, calibration: dict, output: Path) -> None:
    if output.suffix != ".usda" or output.exists():
        raise ValueError("Choose a new .usda output file")
    base = base.resolve(strict=True)
    output = output.parent.resolve(strict=True) / output.name
    # Keep the sidecar beside its final path so relative assets resolve even while validating edits.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".usda", dir=output.parent, delete=False) as temp:
        temp.write("#usda 1.0\n")
    temp_path = Path(temp.name)
    try:
        layer = Sdf.Layer.FindOrOpen(str(temp_path))
        source = Sdf.Layer.FindOrOpen(str(base))
        layer.subLayerPaths = [os.path.relpath(base, output.parent)]
        for key in source.pseudoRoot.ListInfoKeys():
            if key not in ("subLayers", "subLayerOffsets"):
                value = source.pseudoRoot.GetInfo(key)
                if key == "customLayerData":
                    value.get("omni_layer", {}).pop("authoring_layer", None)
                layer.pseudoRoot.SetInfo(key, value)
        stage = Usd.Stage.Open(layer)
        rig = cameras(stage)
        if not calibration or calibration.keys() - rig.keys():
            raise ValueError(f"Expected camera names from {', '.join(rig)}")
        for name, data in calibration.items():
            author(rig[name], data)
        if not layer.Save():
            raise OSError(f"Could not save {output}")
        os.link(temp_path, output)
    finally:
        temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=BASE_STAGE)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("inspect")
    scan.add_argument("--native-session", type=Path)
    edit = sub.add_parser("apply")
    edit.add_argument("calibration", type=Path)
    edit.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect(Usd.Stage.Open(str(args.stage.resolve())), args.native_session)
        print(json.dumps(result, indent=2))
    else:
        apply(args.stage, json.loads(args.calibration.read_text())["cameras"], args.output)
        print(args.output)


if __name__ == "__main__":
    main()
