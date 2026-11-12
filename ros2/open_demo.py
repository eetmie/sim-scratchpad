"""Kit startup script: open demo_env.usd, run ros2_bridge.py, press Play.

scripts/sim-scratchpad/ros2/run_isaacsim.sh --exec scripts/sim-scratchpad/ros2/open_demo.py

The viewport window is hidden, so it doesn't render (RViz is the view); that saves a full RTX render per frame.
The cameras still render. EXCAVATOR_VIEWPORT=1 keeps it; bring it back at runtime with Window > Viewport > Viewport 1.
"""

import asyncio
import os
from pathlib import Path

import omni.kit.app
import omni.timeline
import omni.usd

HERE = os.path.dirname(os.path.abspath(__file__))


async def hide_viewport():
    """Hide the viewport window and stop its rendering.

    Kit's viewport window code sets updates_enabled = visible a few frames after load (and on dock changes), so
    only a hidden window stays off. The window can also appear late at startup: keep re-asserting for ~5 s.
    """
    from omni.kit.viewport.utility import get_active_viewport_window

    app, hidden = omni.kit.app.get_app(), False
    for _ in range(300):
        window = get_active_viewport_window()
        if window is not None and (window.visible or window.viewport_api.updates_enabled):
            window.visible = False
            window.viewport_api.updates_enabled = False
            hidden = True
        await app.next_update_async()
    if hidden:
        print("[open_demo] Viewport hidden; Window > Viewport > Viewport 1 brings it back.")


async def open_demo():
    await omni.usd.get_context().open_stage_async(os.path.join(HERE, "..", "demo_env.usd"))
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()
    bridge = os.path.join(HERE, "ros2_bridge.py")
    exec(compile(Path(bridge).read_text(), bridge, "exec"), globals())
    omni.timeline.get_timeline_interface().play()
    if os.environ.get("EXCAVATOR_VIEWPORT") != "1":
        await hide_viewport()


asyncio.ensure_future(open_demo())
