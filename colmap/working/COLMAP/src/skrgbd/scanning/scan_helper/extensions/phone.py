from collections import defaultdict

from skrgbd.devices.phone import HuaweiPhone

from skrgbd.utils.logging import logger
from skrgbd.scanning.scan_helper.extensions.task import Task, log_context


class _Phone:
    def __init__(self, phone: HuaweiPhone, ir_switch, scene_dir):
        self.phone = phone
        self.name = phone.name
        self.ir_switch = ir_switch
        self.scene_dir = scene_dir
        self.phone_id = None

        self.light_delay = defaultdict(lambda: 0.)
        for light in 'flash', 'soft_left', 'ambient':
            # delay for auto white balance, depends on the order of the light sources
            self.light_delay[light] = .3

    def ir_off(self):
        desc = f'Turn off {self.phone.name} ({self.phone_id}) IR'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.ir_switch.off()
        return Task(target, desc)

    def ir_on(self):
        desc = f'Turn on {self.phone.name} ({self.phone_id}) IR'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.ir_switch.on()
        return Task(target, desc)

    def save_ir_depth(self, img_name):
        desc = f'Save {self.phone.name} ({self.phone_id}) depth'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.phone.save_image(
                f'{self.scene_dir}/{self.phone.name}/{img_name}', modalities={'depth', 'ir'}, blocking=False)
        return Task(target, desc)

    def save_photo(self, img_name):
        desc = f'Shoot with {self.phone.name} ({self.phone_id})'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            return self.phone.save_image(
                f'{self.scene_dir}/{self.phone.name}/{img_name}', modalities={'image'}, blocking=False)
        return Task(target, desc)

    def setup(self, light, preset):
        def target(task):
            return self.phone.setup(light, preset)
        return Task(target, f'Setup {self.phone.name} for {preset} at {light}')
