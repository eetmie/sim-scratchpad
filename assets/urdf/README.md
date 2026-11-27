# Excavator URDF

`excavator.urdf` and `meshes/` are shared by ROS visualization, robot self-filtering,
and the Newton sensor/soil demo. Mesh filenames are relative to this directory;
`ros2/robot_boxes.py` resolves local file URIs for ROS at runtime.

Regenerate from `demo_env.usd` using `ros2/export_urdf.py` in the Isaac Sim
environment. This replaces the exported mesh files. Check the resulting geometry
with `ros2/check_urdf_fk.py` and run the tests described in
[DEVELOPING.md](../../DEVELOPING.md).

The model derives from Isaac-hydraulic-actuator. Its
[Apache 2.0 license](../excavator_bucket.LICENSE) is retained alongside the USD
source asset. See the [repository README](../../README.md) for upstream attribution.
