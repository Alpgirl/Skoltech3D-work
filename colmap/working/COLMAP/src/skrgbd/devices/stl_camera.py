from abc import ABC, abstractmethod
from time import sleep

from toupcam.camera import ToupCamCamera

from skrgbd.devices.camera import Camera
from skrgbd.utils.logging import logger


class STLCamera(ToupCamCamera, Camera, ABC):
    @property
    @abstractmethod
    def serial(self): ...

    @property
    @abstractmethod
    def name(self): ...

    # Setup procedures
    # ----------------
    def __init__(self):
        Camera.__init__(self)
        self.exposure = -1

    def __del__(self):
        self.stop()

    def start(self):
        logger.debug(f'{self.name}: Start')
        ToupCamCamera.__init__(self, resolution=0, bits=8)
        if self.get_serial() != self.serial:
            raise RuntimeError('Wrong camera')
        self.open()
        sleep(1)
        self._set_image_settings()
        logger.debug(f'{self.name}: Start DONE')

    def stop(self):
        logger.debug(f'{self.name}: Stop')
        self.close()
        logger.debug(f'{self.name}: Stop DONE')

    def setup(self, light, preset):
        r"""Sets up the camera for the light source with the specific imaging preset.

        Parameters
        ----------
        light : {'room_lights'}
        preset : {'calibration'}
        """
        logger.debug(f'{self.name}: Setup for {preset} at {light}')
        if preset == 'calibration':
            if light == 'room_lights':
                exposure = 400_000
            else:
                raise NotImplemented
        else:
            raise NotImplemented
        self.set_exposure(exposure)
        logger.debug(f'{self.name}: Setup for {preset} at {light} DONE')

    def set_exposure(self, exposure):
        r"""
        Parameters
        ----------
        exposure : int
            Exposure in us.
        """
        if exposure == self.exposure:
            return
        self.exposure = exposure
        self.set_exposure_time(exposure)

    def _set_image_settings(self):
        logger.debug(f'{self.name}: Setting image settings')
        self.set_brightness(0)
        self.set_contrast(0)
        self.set_gamma(100)
        self.set_hue(0)
        self.set_saturation(128)
        self.set_auto_exposure(False)
        logger.debug(f'{self.name}: Setting image settings DONE')

    # Scanning procedures
    # -------------------
    def snap_frame(self, equalize_ir=False, compressed=False):
        r"""
        Returns
        -------
        frame : dict
            image : np.ndarray
                of shape [height, width]
        """
        logger.debug(f'{self.name}: Taking image')
        image = self.get_image_data()
        image = image[::-1]
        logger.debug(f'{self.name}: Taking image DONE')
        return dict(image=image)

    # IO procedures
    # -------------
    def save_rgb_calib(self, root_dir, image_name, frame):
        path = root_dir / f'{self.name}'
        path.mkdir(parents=True, exist_ok=True)
        path = path / f'{image_name}'
        self.save_rgb(path, frame)

    # Streaming procedures
    # --------------------
    def start_streaming(self, modality, im_slice=None, figsize_base=5, figsize=None, ticks=True, frames=True, name=True):
        modality = 'image'
        if im_slice is None:
            im_slice = slice(None, None, None)
            if figsize is None:
                figsize = (figsize_base, figsize_base * 1536 / 2048)
        else:
            im = self.snap_frame()[modality][im_slice]
            h, w = im.shape[:2]
            figsize = (figsize_base, figsize_base * h / w)

        kwargs = dict(cmap='gray', vmin=0, vmax=255)
        get_image = lambda: self.snap_frame()[modality][im_slice]
        return super().start_streaming(modality, get_image, figsize, ticks, frames, name, **kwargs)


class RightSTLCamera(STLCamera):
    serial = b'TP2006110942175807DA509CDB45C7C'
    name = 'stl_right'
