"""Sensor contract shared by the sim (ros2_bridge.py) and the ROS side (launch files). Plain Python, no ROS imports.

The real robot (kaivuriprokkis) publishes the same topics and frames; only the data source differs.
Camera frames are depth (IR1) frames in optical convention: x right, y down, z forward.
"""

# Depth cameras: D435i, emitter off (passive stereo), 640 x 480, Z16 in mm, depth aligned to IR1.
# (name, parent URDF link, xyz m, quaternion xyzw, fx, fy)
DEPTH_CAMERAS = [
    # From ~/depth-live-bench/sand-calibration.json via kaivuri_bridge.camera_transforms (real robot, 2026-11-12).
    (
        "cabin",
        "upper_carriage",
        (0.1504791, 0.1234607, 0.1269264),
        (-0.6331445, 0.6464634, -0.3316019, 0.2669331),
        384.375977,
        384.375977,
    ),
    # shortcut: placeholder at the sim's cam_boom mount; the real arm calibration puts it 0.74 m off the arm
    # (the bridge drops cabin_shift_xy_m). Replace once the arm calibration is fixed.
    (
        "arm",
        "tilt_boom",
        (0.04948515, 0.0, -0.03343606),
        (0.545621, -0.545621, 0.4497752, -0.4497752),
        386.54718,
        386.54718,
    ),
]
DEPTH_SIZE = (640, 480)
# The camera prims in demo_env.usd (under the parent link above). The bridge re-applies the pose and lens from
# DEPTH_CAMERAS at startup, so this file stays the one source of truth for the sim and its TF frames.
CAMERA_PRIMS = {"cabin": "cam_cabin", "arm": "cam_boom"}
# shortcut: principal point at the image centre; the real ones are ~5 px off (325.5, 236.2). Fine while the sim
# publishes a matching camera_info; set USD aperture offsets if sim/real pixels must line up exactly.
# Robot calibration (sand_height.py calibrate -> sand-calibration.json): recorder key per camera, and the cameras
# whose published image is turned 180 deg (the depth correction is fitted over the physical sensor pixels).
CALIBRATION_KEYS = {"cabin": "cam2", "arm": "cam1"}
ROTATED_180_ON_ROBOT = ("arm",)
CAMERA_HZ = 30.0  # RGB images (the robot's camera target)
DEPTH_HZ = 5.0  # depth images, and so the height map (the robot's target)
DEPTH_MAX_M = 0.9  # the real sand_height.py drops points beyond 0.8-0.9 m


def camera_frame(name):
    return f"{name}_camera"


def camera_topic(name, leaf):
    """leaf: image_raw (rgb8), depth/image_raw (16UC1 mm), depth/camera_info."""
    return f"/excavator/cameras/{name}/{leaf}"


# Lidars (optional, EXCAVATOR_LIDARS=1). Points are published in these frames, self hits included, like a real lidar.
# (name, parent URDF link, xyz m, quaternion xyzw): the USD prim poses in demo_env.usd.
LIDARS = [
    ("lidar_cabin", "upper_carriage", (0.1138097, 0.0812657, 0.1700001), (0.0, 0.2907022, 0.0, 0.9568136)),
    ("lidar_boom", "tilt_boom", (0.04948515, 0.0, -0.03343606), (0.0, 0.0958458, 0.0, 0.9953962)),
]


# Commands in: the robot's excavator-vla contract (kaivuri-ros-fit adapter). std_msgs/Float32MultiArray
# [slew, lift, tilt, scoop], normalized valve demand in [-1, 1]; a stream older than 150 ms means stop.
COMMAND_TOPIC = "/excavator/command"
COMMAND_TIMEOUT_S = 0.15
COMMAND_JOINTS = ("revolute_carriage", "revolute_lift", "revolute_tilt", "revolute_tool")  # URDF names of AXES
# URDF joint direction of +command: the robot's native_angle_sign (positive demand raises the normalized angle).
# shortcut: the robot's command signs are not commissioned yet; flip here if a real axis turns out reversed.
COMMAND_SIGN = (1, -1, -1, 1)
COMMAND_SPEED_DEG_S = (15.0, 10.0, 12.5, 20.0)  # sim joint speed at |command| = 1 (the old gamepad teleop's)

# Joint angles out: /excavator/joint_angles, std_msgs/Float32MultiArray [slew, lift, tilt, scoop] in [-1, 1]
# (kaivuri_bridge.py: native = URDF - URDF_OFFSET, v = 2 (sign * native - min) / (max - min) - 1;
# bounds from kaivuri.profile.json).
JOINT_ANGLES_TOPIC = "/excavator/joint_angles"
AXES = ("slew", "lift", "tilt", "scoop")
URDF_OFFSET = (0.0, 0.0, -0.02813, 0.00355)
NATIVE_ANGLE_SIGN = (1, -1, -1, 1)
JOINT_ANGLE_MIN = (-3.141592653589793, -0.5235987755982988, -2.5132741228718345, -2.0943951023931953)
JOINT_ANGLE_MAX = (3.141592653589793, 0.8726646259971648, -0.4886921905584123, 0.3490658503988659)

# Height map for the VLA's vision encoder: FastDEM's latest scan seen from above, in the swinging cabin frame
# (so it turns with the camera). Row 0 = far edge (+x), column 0 = left edge (+y), like a top-down photo.
HEIGHTMAP_TOPIC = "/excavator/heightmap"  # 32FC1, z in HEIGHTMAP_FRAME [m], NaN = not seen
HEIGHTMAP_IMAGE_TOPIC = "/excavator/heightmap/image"  # mono8, 1..255 over HEIGHTMAP_Z_M, 0 = not seen
HEIGHTMAP_FRAME = "upper_carriage"
HEIGHTMAP_X_M = (0.10, 0.90)
HEIGHTMAP_Y_M = (-0.40, 0.40)
HEIGHTMAP_SIZE = 64  # 12.5 mm cells
HEIGHTMAP_Z_M = (-0.30, 0.00)  # flat ground (~-0.15 in upper_carriage) is mid-gray
