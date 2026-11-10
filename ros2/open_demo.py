"""Kit startup script: open demo_env.usd, run ros2_bridge.py, press Play.

scripts/sim-scratchpad/ros2/run_isaacsim.sh --exec scripts/sim-scratchpad/ros2/open_demo.py
"""

import asyncio
import os
from pathlib import Path

import omni.kit.app
import omni.timeline
import omni.usd

HERE = os.path.dirname(os.path.abspath(__file__))


async def open_demo():
    await omni.usd.get_context().open_stage_async(os.path.join(HERE, "..", "demo_env.usd"))
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()
    bridge = os.path.join(HERE, "ros2_bridge.py")
    exec(compile(Path(bridge).read_text(), bridge, "exec"), globals())
    omni.timeline.get_timeline_interface().play()


asyncio.ensure_future(open_demo())
