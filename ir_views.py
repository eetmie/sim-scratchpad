"""Small calibrated grayscale views; depth products retain their own resolution."""

import copy

import numpy as np


def scaled_camera(info, width):
    """Resize the whole image, preserving FOV, aspect ratio and pixel centres."""
    info = copy.deepcopy(info)
    intr = info["intrinsics"]
    height = max(1, round(intr["height"] * width / intr["width"]))
    sx, sy = width / intr["width"], height / intr["height"]
    intr.update(
        width=width,
        height=height,
        fx=intr["fx"] * sx,
        fy=intr["fy"] * sy,
        cx=(intr["cx"] + 0.5) * sx - 0.5,
        cy=(intr["cy"] + 0.5) * sy - 0.5,
    )
    return info


class IRViews:
    def __init__(self, stage, width=224):
        from camera_calibration.usd_cameras import author, cameras, describe

        import omni.replicator.core as rep
        from pxr import UsdGeom

        self.products, self.images = {}, {}
        for name, source in cameras(stage).items():
            info = scaled_camera(describe(source), width)
            info["prim"] = str(source.GetPrim().GetParent().GetPath()) + f"/preview_ir_{name}"
            camera = UsdGeom.Camera.Define(stage, info["prim"])
            camera.CreateClippingRangeAttr().Set(source.GetClippingRangeAttr().Get())
            author(camera, info)
            intr = info["intrinsics"]
            product = rep.create.render_product(info["prim"], (intr["width"], intr["height"]), force_new=True)
            color = rep.annotators.get("LdrColor", device="cpu")
            color.attach(product.path)
            self.products[name] = (info, product, color)

    def read(self):
        from camera_calibration.capture import ir_image

        images = {}
        for name, (info, _, color) in self.products.items():
            rgba = np.asarray(color.get_data())
            intr = info["intrinsics"]
            if rgba.shape != (intr["height"], intr["width"], 4):
                return False
            # A camera can legitimately face a uniform surface during teleoperation.
            if rgba[..., :3].min() == rgba[..., :3].max():
                images[name] = np.full(rgba.shape[:2], rgba[0, 0, 0], dtype=np.uint8)
            else:
                images[name] = ir_image(rgba, info)
        self.images = images
        return True

    def close(self):
        for _, product, color in self.products.values():
            color.detach(product.path)
            product.destroy()
