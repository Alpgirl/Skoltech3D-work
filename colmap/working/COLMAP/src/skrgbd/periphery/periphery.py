from skrgbd.periphery.arduino import Arduino
from skrgbd.periphery.flashlight import Flashlight


class Periphery:
    def __init__(self, phones):
        self._arduino = Arduino()
        self._flash = Flashlight(phones)

        self.kinect_emitter = self._arduino.kinect_emitter
        self.kinect_light_filter = self._arduino.kinect_light_filter
        self.left_phone_ir = self._arduino.left_phone_ir
        self.right_phone_ir = self._arduino.right_phone_ir

        self.lights = dict(
            flash=self._flash,
            soft_left=self._arduino.soft_left,
            soft_right=self._arduino.soft_right,
            soft_top=self._arduino.soft_top,
            hard_left_bottom_far=self._arduino.hard_left_bottom_far,
            hard_left_bottom_close=self._arduino.hard_left_bottom_close,
            hard_left_top_close=self._arduino.hard_left_top_close,
            hard_right_top_far=self._arduino.hard_right_top_far,
            hard_right_top_close=self._arduino.hard_right_top_close,
            hard_left_top_far=self._arduino.hard_left_top_far,
            hard_right_bottom_close=self._arduino.hard_right_bottom_close,
            ambient=self._arduino.ambient_led,
            ambient_low=self._arduino.ambient_led_low,
        )
        for k, v in self.lights.items():
            setattr(self, k, v)

    def lights_off(self):
        self.on_only('')

    def on_only(self, light):
        for switch_name, switch in self.lights.items():
            if switch_name != light:
                switch.off()
        if light in self.lights:
            self.lights[light].on()

    def off(self):
        self.on_only('')
