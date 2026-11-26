"""Exercise real pyglet event binding for the Ultimate 2C Bluetooth layout."""

import pyglet
import pytest

pyglet.options["shadow_window"] = False
import pyglet.input
from pyglet.input import controller as mappings
from pyglet.input.base import Device
from pyglet.input.linux import evdev
from teleop import Teleop

pytestmark = pytest.mark.kitless
GUID = "05000000c82d00001b30000001000000"


class Ultimate2C(Device):
    """In-memory device with the connected pad's controls; pyglet handles events."""

    ff_types = ()

    def __init__(self):
        super().__init__(None, "8BitDo Ultimate 2C Wireless")
        self.controls = []
        for code, name in ((0, "x"), (1, "y"), (2, "z"), (5, "rz"), (9, None), (10, None)):
            axis = evdev.EvdevAbsoluteAxis(name, 0, 255)
            axis.event_type, axis.event_code = evdev.EV_ABS, code
            self.controls.append(axis)
        for code, name in ((16, "hat_x"), (17, "hat_y")):
            axis = evdev.EvdevAbsoluteAxis(name, -1, 1)
            axis.event_type, axis.event_code = evdev.EV_ABS, code
            self.controls.append(axis)
        for code in range(304, 320):
            button = evdev.EvdevButton(None)
            button.event_type, button.event_code = evdev.EV_KEY, code
            self.controls.append(button)

    def get_guid(self):
        return GUID

    def get_controls(self):
        return self.controls


class NewtonGLVisualizer:
    def set_camera_keys_enabled(self, enabled):
        pass

    def is_key_down(self, key):
        return False


@pytest.mark.parametrize("user_mapping", [False, True])
def test_both_sticks_drive_all_joints_and_triggers_stay_separate(monkeypatch, user_mapping):
    """Z/RZ must reach bucket/boom, while an explicit user mapping keeps priority."""
    monkeypatch.setattr(mappings, "mapping_list", [])
    if user_mapping:
        mappings.add_mappings_from_string(
            f"{GUID},Custom,leftx:a0,lefty:a1,rightx:a3,righty:a2,lefttrigger:a5,righttrigger:a4,"
        )
    device = Ultimate2C()
    monkeypatch.setattr(pyglet.input, "get_controllers", lambda: [evdev._create_controller(device)])
    teleop = Teleop(
        {
            "slew": ("left_x", "D", "A", -15),
            "arm": ("left_y", "W", "S", -12.5),
            "bucket": ("right_x", "L", "J", -20),
            "boom": ("right_y", "I", "K", 10),
        },
        [NewtonGLVisualizer()],
    )
    try:
        for axis, value in zip(device.controls[:6], (255, 0, 0, 255, 0, 0)):
            axis.value = value
        assert teleop.axes() == dict(slew=1, arm=1, bucket=1 if user_mapping else -1, boom=1 if user_mapping else -1)
        for axis in device.controls[:4]:
            axis.value = 127
        # The physical triggers are ABS_GAS / ABS_BRAKE, not Z / RZ.
        device.controls[4].value = device.controls[5].value = 255
        assert teleop.axes() == dict(slew=0, arm=0, bucket=0, boom=0)
        assert teleop.source.controller.lefttrigger == 1
        assert teleop.source.controller.righttrigger == 1
    finally:
        teleop.source.controller.close()
