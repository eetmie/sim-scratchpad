"""Gamepad + keyboard input for both front ends.

Kit: carb input from the Kit app window.
Newton viewer: the viewer's keyboard, and the first game controller through pyglet (SDL mapping),
pumped here because the Newton viewer does not run pyglet's event loop.
Both report the same stick axes (-1..1, right/up positive) and buttons.
"""

import math

DEADZONE = 0.20
STICKS = ("left_x", "left_y", "right_x", "right_y")

# Bluetooth Ultimate 2C uses Z/RZ for its right stick and GAS/BRAKE for
# triggers. Pyglet's generic Linux fallback assumes RX/RY and loses the stick.
CONTROLLER_MAPPINGS = (
    "05000000c82d00001b30000001000000,8BitDo Ultimate 2C Wireless,"
    "a:b0,b:b1,x:b4,y:b3,back:b10,start:b11,guide:b12,"
    "leftshoulder:b6,rightshoulder:b7,leftstick:b13,rightstick:b14,"
    "leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a5,righttrigger:a4,"
    "dpup:h0.1,dpright:h0.2,dpdown:h0.4,dpleft:h0.8,platform:Linux,",
)


def apply_deadzone(value):
    """Zero small stick noise, then rescale so the output still spans -1..1."""
    magnitude = (abs(value) - DEADZONE) / (1.0 - DEADZONE)
    return 0.0 if magnitude <= 0.0 else math.copysign(min(magnitude, 1.0), value)


class _Source:
    name = "none"

    def stick(self, axis):
        return 0.0

    def button(self, name):
        return False

    def key(self, name):
        return False


class KitInput(_Source):
    name = "Kit"

    def __init__(self, window):
        import carb
        import carb.input

        self.carb_input = carb.input
        self.input = carb.input.acquire_input_interface()
        self.gamepad = window.get_gamepad(0)
        self.keyboard = window.get_keyboard()
        # Otherwise the same sticks also fly the viewport camera.
        carb.settings.get_settings().set("/persistent/app/omniverse/gamepadCameraControl", False)
        name = self.input.get_gamepad_name(self.gamepad) if self.gamepad else ""
        self.description = f"Kit gamepad 0: {name or 'not found (keyboard still works)'}"

    def _pad(self, name):
        if not self.gamepad:
            return 0.0
        return float(self.input.get_gamepad_value(self.gamepad, getattr(self.carb_input.GamepadInput, name)))

    def stick(self, axis):
        side, direction = axis.split("_")
        side = side.upper()
        if direction == "x":
            return self._pad(f"{side}_STICK_RIGHT") - self._pad(f"{side}_STICK_LEFT")
        return self._pad(f"{side}_STICK_UP") - self._pad(f"{side}_STICK_DOWN")

    def button(self, name):
        return self._pad(name.upper()) > 0.5

    def key(self, name):
        return self.input.get_keyboard_value(self.keyboard, getattr(self.carb_input.KeyboardInput, name.upper())) > 0


class NewtonViewerInput(_Source):
    name = "Newton viewer"

    def __init__(self, visualizer):
        self.visualizer = visualizer
        visualizer.set_camera_keys_enabled(False)  # WASD drive the excavator; the mouse still moves the camera
        self.controller = None
        try:
            import pyglet.input
            from pyglet.input.controller import add_mappings_from_string, get_mapping

            for mapping in CONTROLLER_MAPPINGS:
                if get_mapping(mapping.partition(",")[0]) is None:
                    add_mappings_from_string(mapping)
            controllers = pyglet.input.get_controllers()
            if controllers:
                self.controller = controllers[0]
                self.controller.open()
        except Exception as error:  # no permission on /dev/input, no device, ...
            print(f"[Teleop] Game controller unavailable: {error}")
            self.controller = None
        name = getattr(getattr(self.controller, "device", None), "name", "") if self.controller else ""
        self.description = f"pyglet controller: {name or 'not found (keyboard still works)'}"

    def poll(self):
        device = getattr(self.controller, "device", None)
        if device is None:
            return
        try:
            while device.poll():
                device.select()
        except (AttributeError, OSError):
            pass

    def stick(self, axis):
        if self.controller is None:
            return 0.0
        value = {
            "left_x": self.controller.leftx,
            "left_y": -self.controller.lefty,  # pyglet's raw y is positive down
            "right_x": self.controller.rightx,
            "right_y": -self.controller.righty,
        }[axis]
        return float(value)

    def button(self, name):
        return bool(self.controller is not None and getattr(self.controller, name.lower(), False))

    def key(self, name):
        return self.visualizer.is_key_down(name.lower())


class Teleop:
    """Rate commands per joint plus edge-triggered buttons, from whichever front end is running."""

    def __init__(self, joint_axes, visualizers):
        """``joint_axes``: joint -> (stick axis, key +, key -, speed deg/s)."""
        self.joint_axes = joint_axes
        self.previous = {}
        self.source = _Source()
        names = {type(v).__name__: v for v in visualizers}
        try:
            import omni.appwindow

            window = omni.appwindow.get_default_app_window() if "KitVisualizer" in names else None
        except ImportError:
            window = None
        if window is not None:
            self.source = KitInput(window)
        elif "NewtonGLVisualizer" in names:
            self.source = NewtonViewerInput(names["NewtonGLVisualizer"])
        print(f"[Teleop] {getattr(self.source, 'description', 'No window (headless); the arm holds its pose.')}")

    def axes(self):
        """Joint name -> command in -1..1 (deadzoned stick, or full-rate key)."""
        if hasattr(self.source, "poll"):
            self.source.poll()
        commands = {}
        for joint, (axis, key_pos, key_neg, _) in self.joint_axes.items():
            stick = apply_deadzone(self.source.stick(axis))
            commands[joint] = stick or (float(self.source.key(key_pos)) - float(self.source.key(key_neg)))
        return commands

    def pressed(self, button, key):
        """Rising edge of a gamepad button or key."""
        down = self.source.button(button) or self.source.key(key)
        was = self.previous.get(button, False)
        self.previous[button] = down
        return down and not was
