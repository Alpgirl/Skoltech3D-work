from skrgbd.periphery.periphery import Periphery

from skrgbd.utils.logging import logger
from skrgbd.scanning.scan_helper.extensions.task import Task, log_context


class _Lights:
    def __init__(self, periphery: Periphery):
        self.periphery = periphery
        self.off = periphery.lights_off

    def setup_light(self, light):
        desc = f'Setup light {light}'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            self.periphery.on_only(light)
            logger.debug(f'{log_context}: {desc} DONE')
        return Task(target, desc)
