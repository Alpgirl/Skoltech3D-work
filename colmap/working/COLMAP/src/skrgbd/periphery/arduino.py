from pathlib import Path
from itertools import permutations
from threading import RLock
from serial import Serial


class Arduino:
    def __init__(self, device=None):
        if device is None:
            device = next(Path('/dev').glob('ttyACM*'))
        self.serial = Serial(str(device))

        self.kinect_emitter = Switch(self, 0)
        self.kinect_light_filter = Switch(self, 12)
        self.left_phone_ir = Switch(self, 22)
        self.right_phone_ir = Switch(self, 23)
        self.ambient_led = Switch(self, 1)
        self.ambient_led_low = Switch(self, 21)
        self.soft_right = Switch(self, 2)
        self.soft_left = Switch(self, 3)
        self.soft_top = Switch(self, 4)

        for i, pos in enumerate(start=5, iterable=[
            ('left', 'bottom', 'far'),
            ('left', 'top', 'far'),
            ('right', 'top', 'far'),
            ('left', 'bottom', 'close'),
            ('right', 'bottom', 'close'),
            ('left', 'top', 'close'),
            ('right', 'top', 'close'),
        ]):
            switch = Switch(self, i)
            for p in permutations(pos):
                setattr(self, f'hard_{p[0]}_{p[1]}_{p[2]}', switch)
        self.lock = RLock()

        if self.serial.readline() != b'ready\r\n':
            raise RuntimeError('Failed to initialize')

    def __del__(self):
        self.serial.close()

    def id_set(self, i, value):
        r"""Sends on/off to the peripheral device with id.

        Parameters
        ----------
        i : int
        value : {0, 1}

        Notes
        -----
        Encodes the id and the value into a single byte.
        """
        self.lock.acquire()
        self.serial.write(int(i * 2 + value).to_bytes(1, 'big'))
        self.serial.readline()
        self.lock.release()


class Switch:
    def __init__(self, arduino, i):
        self.arduino = arduino
        self.id = i

    def on(self):
        self.arduino.id_set(self.id, 1)

    def off(self):
        self.arduino.id_set(self.id, 0)
