from collections import defaultdict

from skrgbd.devices.kinect import Kinect
from skrgbd.periphery.periphery import Periphery

from skrgbd.utils.logging import logger
from skrgbd.scanning.scan_helper.extensions.task import Task, log_context


class _Kinect:
    def __init__(self, kinect: Kinect, periphery: Periphery, scene_dir, trajectory):
        self.kinect = kinect
        self.name = kinect.name
        self.periphery = periphery
        self.scene_dir = scene_dir
        self.trajectory = trajectory

        self.light_delay = defaultdict(lambda: 1)  # delay for awb and autoexposure

    def ir_off(self):
        desc = f'Close Kinect IR'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.periphery.kinect_emitter.off()
        return Task(target, desc)

    def ir_on(self):
        desc = f'Open Kinect IR'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.periphery.kinect_emitter.on()
        return Task(target, desc)

    def save_photo(self, img_name):
        desc = f'Shoot with {self.kinect.name}'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.kinect.save_image(
                f'{self.scene_dir}/{self.kinect.name}/{img_name}', modalities={'image'}, blocking=False, compressed=True)
        return Task(target, desc)

    def save_ir_depth(self, img_name):
        desc = f'Save Kinect depth'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.kinect.save_image(
                f'{self.scene_dir}/{self.kinect.name}/{img_name}', modalities={'depth', 'ir'}, blocking=False, compressed=True)
        return Task(target, desc)

    def setup(self, light, preset):
        desc = f'Setup {self.kinect.name} for {light}'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            if self.trajectory == 'spheres':
                if light in {'ambient_low', 'flash', 'hard_left_bottom_far'}:
                    self.periphery.kinect_light_filter.off()
                else:
                    self.periphery.kinect_light_filter.on()
            else:
                self.periphery.kinect_light_filter.off()
        return Task(target, desc)
