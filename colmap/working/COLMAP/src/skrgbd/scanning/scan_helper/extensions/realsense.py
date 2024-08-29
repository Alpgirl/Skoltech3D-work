from collections import defaultdict

from skrgbd.devices.realsense import RealSense

from skrgbd.utils.logging import logger
from skrgbd.scanning.scan_helper.extensions.task import Task, log_context


class _RealSense:
    def __init__(self, realsense: RealSense, scene_dir):
        self.realsense = realsense
        self.name = realsense.name
        self.scene_dir = scene_dir

        self.light_delay = defaultdict(lambda: .5)  # delay for not instant light switching
        self.light_delay['ambient_low'] = 1  # don't ask why
        self.light_delay['ambient'] = 1.5  # at high exposures real sense captures the light from the previous light source

    def save_all(self, img_name):
        desc = f'Shoot with {self.realsense.name}'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.realsense.save_image(
                f'{self.scene_dir}/{self.realsense.name}/{img_name}',
                modalities={'image', 'depth', 'ir', 'ir_right'}, blocking=False, compressed=True)
        return Task(target, desc)

    def setup(self, light, preset):
        def target(task):
            return self.realsense.setup(light, preset)
        return Task(target, f'Setup {self.realsense.name} for {preset} at {light}')
