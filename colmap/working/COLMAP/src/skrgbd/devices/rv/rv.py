from pathlib import Path
from time import sleep

from PIL import Image
import pyautogui
import pyscreenshot as ImageGrab
import numpy as np
import torch

from skrgbd.utils.logging import logger


class RVGui:
    name = 'RVGui'
    screen_w = 1300
    screen_h = 1080

    def __init__(self, in_vm=False):
        if in_vm:
            self._in_vm = True
            self.x_shift = 0
            self.y_shift = 0
            self.img_dir = r'Z:\sk_robot_rgbd_data\src\skrgbd\devices\rv\img'
        else:
            self._in_vm = False
            self.x_shift = 1920 * 2
            self.y_shift = 0
            self.img_dir = '/home/universal/Downloads/dev.sk_robot_rgbd_data/src/skrgbd/devices/rv/img'
        img_dir = Path(self.img_dir)
        self.scan_center_window = load_img(img_dir / 'scan_center_window.png')
        self.home_inactive = load_img(img_dir / 'home_inactive.png')
        self.home_inactive_v2 = load_img(img_dir / 'home_inactive_v2.png')
        self.home_active = load_img(img_dir / 'home_active.png')
        self.new_project_basic = load_img(img_dir / 'new_project_basic.png')
        self.enter_project_name = load_img(img_dir / 'enter_project_name.png')
        self.enter_project_name_shift_y = 20
        self.texture_on = load_img(img_dir / 'texture_on.png')
        self.texture_off = load_img(img_dir / 'texture_off.png')
        self.continue_btn = load_img(img_dir / 'continue.png')
        self.dont_save = load_img(img_dir / 'dont_save.png')
        self.scanning_tab_active = load_img(img_dir / 'scanning_tab.png')
        self.black_inactive = load_img(img_dir / 'black_inactive.png')
        # self.black_active = load_img(img_dir / 'black_active.png')
        self.start_scanning = load_img(img_dir / 'start_scanning.png')
        self.calculation = load_img(img_dir / 'calculation.png')
        self.please_wait_end = load_img(img_dir / 'please_wait_end.png')
        self.processing_uptab_inactive = load_img(img_dir / 'processing.png')
        self.export_tab_inactive = load_img(img_dir / 'export_inactive.png')
        self.object_name = load_img(img_dir / 'object_name.png')
        self.object_name_to_meshes_shift_y = 27
        self.format_dropdown = load_img(img_dir / 'format_dropdown.png')
        self.obj_format = load_img(img_dir / 'obj.png')
        self.export = load_img(img_dir / 'export.png')
        self.export_done = load_img(img_dir / 'export_done.png')
        self.scanning_uptab_inactive = load_img(img_dir / 'scanning_uptab_inactive.png')
        self.scanning_uptab_inactive_v2 = load_img(img_dir / 'scanning_uptab_inactive_v2.png')

    def init_project(self, name):
        if not self._in_vm:
            pyautogui.click(self.locate_img(self.scan_center_window))

        logger.debug(f'{self.name}: Go home')
        if self.locate_img(self.home_active, once=True) is None:
            while True:
                pos = self.locate_img(self.home_inactive, once=True)
                if pos is not None:
                    pyautogui.click(pos)
                    break
                pos = self.locate_img(self.home_inactive_v2, once=True)
                if pos is not None:
                    pyautogui.click(pos)
                    break

        logger.debug(f'{self.name}: Start new project')
        pyautogui.click(self.locate_img(self.new_project_basic))

        logger.debug(f'{self.name}: Name the project')
        x, y = self.locate_img(self.enter_project_name)
        pyautogui.click((x, y + self.enter_project_name_shift_y))
        pyautogui.typewrite(name)

        logger.debug(f'{self.name}: Switch off texturing')
        texture_on = self.locate_img(self.texture_on, once=True)
        if texture_on is not None:
            pyautogui.click(texture_on)

        logger.debug(f'{self.name}: Go scanning')
        pyautogui.click(self.locate_img(self.continue_btn))

        logger.debug(f'{self.name}: Check save')
        while True:
            dont_save = self.locate_img(self.dont_save, once=True)
            if dont_save is not None:
                logger.debug(f'{self.name}: Cancel save')
                pyautogui.click(dont_save)
                break
            start_scanning = self.locate_img(self.start_scanning, once=True)
            if start_scanning is not None:
                logger.debug(f'{self.name}: No save')
                break

    def prepare_for_scanning(self):
        logger.debug(f'{self.name}: Check scanning active')
        if self.locate_img(self.scanning_tab_active) is None:
            raise RuntimeError('Scanning tab not found or inactive')

    def scan(self):
        if not self._in_vm:
            pyautogui.click(self.locate_img(self.scan_center_window))

        black = self.locate_img(self.black_inactive, once=True)
        if black is not None:
            logger.debug(f'{self.name}: Turn off projector')
            pyautogui.click(black)

        logger.debug(f'{self.name}: Scan')
        pyautogui.click(self.locate_img(self.start_scanning))
        pyautogui.move(0, 100)

        logger.debug(f'{self.name}: Wait for scanning')
        self.locate_img(self.calculation)
        logger.debug(f'{self.name}: Wait for scanning DONE')

    def wait_for_scanning_end(self):
        logger.debug(f'{self.name}: Wait for calculation')
        while self.locate_img(self.calculation, once=True) is not None:
            sleep(.3)

        logger.debug(f'{self.name}: Wait for end of scanning')
        while self.locate_img(self.please_wait_end, once=True) is not None:
            sleep(.3)
        logger.debug(f'{self.name}: Wait DONE')

    def export_scans(self):
        if not self._in_vm:
            pyautogui.click(self.locate_img(self.scan_center_window))

        logger.debug(f'{self.name}: Go processing')
        pyautogui.click(self.locate_img(self.processing_uptab_inactive))

        logger.debug(f'{self.name}: Go export')
        pyautogui.click(self.locate_img(self.export_tab_inactive))

        logger.debug(f'{self.name}: Select all scans')
        x, y = self.locate_img(self.object_name)
        pyautogui.click(x, y + self.object_name_to_meshes_shift_y)

        logger.debug(f'{self.name}: Select format')
        pyautogui.click(self.locate_img(self.format_dropdown))

        logger.debug(f'{self.name}: Select OBJ')
        pyautogui.click(self.locate_img(self.obj_format))

        logger.debug(f'{self.name}: Export')
        pyautogui.click(self.locate_img(self.export))

        logger.debug(f'{self.name}: Wait for export')
        pyautogui.click(self.locate_img(self.export_done))
        pyautogui.press('enter')

        logger.debug(f'{self.name}: Go scanning')
        while True:
            pos = self.locate_img(self.scanning_uptab_inactive, once=True)
            if pos is not None:
                pyautogui.click(pos)
                break
            pos = self.locate_img(self.scanning_uptab_inactive_v2, once=True)
            if pos is not None:
                pyautogui.click(pos)
                break

        self.locate_img(self.scanning_tab_active)

    def locate_img(self, img, once=False):
        r"""
        Parameters
        ----------
        img : torch.ByteTensor
            of shape [h, w]
        once : bool

        Returns
        -------
        x : int
        y : int
        """
        h, w = img.shape
        while True:
            screen = ImageGrab.grab(
                bbox=(self.x_shift, self.y_shift, self.x_shift + self.screen_w, self.y_shift + self.screen_h),
                backend='mss', childprocess=False
            )
            screen = torch.from_numpy(np.asarray(screen)[..., 0])
            poses = (screen.unfold(0, h, 1).unfold(1, w, 1) == img).all(-1).all(-1).nonzero()
            if len(poses) == 1:
                y, x = poses[0].tolist()
                break
            elif len(poses) > 1:
                raise RuntimeError(f'Found {len(poses)} instances of the button')
            elif once:
                return None
        x = self.x_shift + x + w // 2
        y = self.y_shift + y + h // 2
        return x, y


def load_img(path):
    return torch.from_numpy(np.array(Image.open(str(path))))
