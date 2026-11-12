"""Run ros2_bridge.py headless (as the Script Editor would) for N seconds. Used by smoke_test.sh."""

import os
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
omni.usd.get_context().open_stage(os.path.join(HERE, "..", "demo_env.usd"))
for _ in range(30):
    app.update()
bridge = os.path.join(HERE, "ros2_bridge.py")
namespace = {"__name__": "__main__"}
exec(compile(Path(bridge).read_text(), bridge, "exec"), namespace)
stage = omni.usd.get_context().get_stage()
assert not stage.GetRootLayer().GetPrimAtPath("/ROS2Bridge"), "bridge graph leaked into demo_env.usd's layer"
omni.timeline.get_timeline_interface().play()
print("[headless_run] playing", flush=True)
end = time.time() + float(sys.argv[1])
while time.time() < end:
    app.update()
print("[headless_run] done", flush=True)
app.close()
