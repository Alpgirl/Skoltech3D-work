from collections import defaultdict

from skrgbd.devices.tis import TisCamera

from skrgbd.utils.logging import logger
from skrgbd.scanning.scan_helper.extensions.task import Task, log_context


class _Tis:
    def __init__(self, tis: TisCamera, scene_dir):
        self.tis = tis
        self.name = tis.name
        self.scene_dir = scene_dir

        self.light_delay = defaultdict(lambda: .3)  # delay for not instant light switching

    def save_photo(self, img_name):
        desc = f'Shoot with {self.tis.name}'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.tis.save_image(f'{self.scene_dir}/{self.tis.name}/{img_name}', blocking=False, compressed=True)
        return Task(target, desc)

    def setup(self, light, preset):
        def target(task):
            self.tis.setup(light, preset)
        return Task(target, f'Setup {self.tis.name} for {preset} at {light}')

    def set_gain(self, gain):
        desc = f'Setup {self.tis.name} with gain {gain}'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            if gain != self.tis.gain:
                self.tis.set_gain(gain)
                self.tis.gain = gain
            logger.debug(f'{log_context}: {desc} DONE')
        return Task(target, desc)
