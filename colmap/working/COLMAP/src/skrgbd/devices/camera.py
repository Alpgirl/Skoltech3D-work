from abc import ABC, abstractmethod
from pathlib import Path
from time import sleep

from PIL import Image
import numpy as np

from skrgbd.utils.frame_widget import FrameWidget
from skrgbd.utils.parallel import get_timestamp, PropagatingThread as Thread
from skrgbd.utils.logging import logger


class Camera(ABC):
    started = None

    def __init__(self):
        self._stop_warmup = False
        self._warmup_thread = None

        self._streaming_widgets = dict()
        self._stop_streaming_widget = dict()
        self._streaming_threads = dict()

    @property
    @abstractmethod
    def name(self): ...

    @abstractmethod
    def start(self):
        r"""Starts the streaming on camera."""
        ...

    @abstractmethod
    def stop(self):
        r"""Stops the streaming on camera."""
        ...

    @abstractmethod
    def setup(self, light, preset):
        r"""Sets up the camera for the light source with the specific imaging preset.

        Parameters
        ----------
        light : str
        preset : str
        """
        ...

    @abstractmethod
    def snap_frame(self, compressed=False):
        r"""Captures images of all modalities of the sensor.

        Parameters
        ----------
        compressed : bool
            If True, returns all modalities in the format compatible with saving to PNG.

        Returns
        -------
        frame : dict
            image : np.ndarray
                of shape [height, width, 3] and dtype uint8.
            depth : np.ndarray
                of shape [height_d, width_d] in millimeters if `compressed` is False.
            ir : np.ndarray
                of shape [height_d, width_d] in [0,1] range if `compressed` is False
        """
        ...

    def save_image(self, path, modalities=None, compressed=False, blocking=True):
        r"""Saves images of the given modalities.

        Parameters
        ----------
        path : str
            Save RGB images to {path}.png, depthmaps to {path}_depth.npy, and IR images to {path}_ir.npy.
            If compressed is True, save depthmaps and IR images to PNG insted.
        modalities : iterable of {'image', 'depth', 'ir'}
        compressed : bool
        blocking : bool
            If False, write the files in background.
        """
        if modalities is None:
            modalities = {'image', 'depth', 'ir'}
        frame = self.snap_frame(compressed=compressed)
        frame = {modality: frame[modality] for modality in modalities if modality in frame}

        def save():
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            if 'image' in frame:
                self.save_rgb(path, frame)
            if 'depth' in frame:
                if compressed:
                    self.save_depth_png(f'{path}_depth', frame)
                else:
                    self.save_depth_npy(f'{path}_depth', frame)
            if 'ir' in frame:
                if compressed:
                    self.save_ir_png(f'{path}_ir', frame)
                else:
                    self.save_ir_npy(f'{path}_ir', frame)

        if blocking:
            save()
        else:
            Thread(target=save).start()

    def save_calib_data(self, root_dir, image_name, modalities, blocking=True):
        r"""Saves images to {root_dir}/{camera_name}_{modality}/{image_name}.png.

        Parameters
        ----------
        root_dir : str
        image_name : str
        modalities : iterable of {'image', 'ir'}
        blocking : bool
            If False, write the files in background.
        """
        frame = self.snap_frame(equalize_ir=True)
        frame = {modality: frame[modality] for modality in modalities}
        root_dir = Path(root_dir)

        def save():
            if 'image' in frame:
                self.save_rgb_calib(root_dir, image_name, frame)
            if 'ir' in frame:
                self.save_ir_calib(root_dir, image_name, frame)

        if blocking:
            save()
        else:
            Thread(target=save).start()

    def save_rgb(self, path, frame):
        image = frame['image']
        path = f'{path}.png'
        logger.debug(f'{self.name}: Saving image to {path}')
        Image.fromarray(image).save(path)

    def save_rgb_calib(self, root_dir, image_name, frame):
        path = root_dir / f'{self.name}_rgb'
        path.mkdir(parents=True, exist_ok=True)
        path = path / f'{image_name}'
        self.save_rgb(path, frame)

    def save_depth_npy(self, path, frame):
        image = frame['depth']
        path = f'{path}.npy'
        logger.debug(f'{self.name}: Saving depth to {path}')
        np.save(path, image)

    def save_depth_png(self, path, frame):
        image = frame['depth']
        path = f'{path}.png'
        logger.debug(f'{self.name}: Saving depth to {path}')
        Image.fromarray(image).save(path)

    def save_ir_npy(self, path, frame):
        image = frame['ir']
        path = f'{path}.npy'
        logger.debug(f'{self.name}: Saving IR to {path}')
        np.save(path, image)

    def save_ir_png(self, path, frame):
        image = frame['ir']
        path = f'{path}.png'
        logger.debug(f'{self.name}: Saving IR to {path}')
        Image.fromarray(image).save(path)

    def save_ir_calib(self, root_dir, image_name, frame):
        path = root_dir / f'{self.name}_ir'
        path.mkdir(parents=True, exist_ok=True)
        path = path / f'{image_name}'

        image = frame['ir']
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        self.save_ir_png(path, {'ir': image})

    def start_warmup(self, shoot_interval=0):
        r"""Keeps the camera warm, taking images in a background thread.
        Call `stop_warmup` to stop.

        Parameters
        ----------
        shoot_interval : float
            Time interval in seconds between subsequent image captures.
        """
        self._stop_warmup = False

        def target():
            prev_shoot_time = 0
            while not self._stop_warmup:
                sleep(max(.01, prev_shoot_time + shoot_interval - get_timestamp()))
                prev_shoot_time = get_timestamp()
                self.snap_frame()
        self._warmup_thread = Thread(target=target)
        self._warmup_thread.start()

    def stop_warmup(self):
        self._stop_warmup = True
        self._warmup_thread.join()

    def start_streaming(self, modality, get_image, figsize, ticks=True, frames=True, name=True, **kwargs):
        if name:
            name = f'{self.name}.{modality}'
        self._streaming_widgets[modality] = FrameWidget(get_image, figsize, ticks, frames, name, **kwargs)
        self._stop_streaming_widget[modality] = False

        def target():
            while not self._stop_streaming_widget[modality]:
                self._streaming_widgets[modality].update()
                sleep(.01)
        self._streaming_threads[modality] = Thread(target=target)
        self._streaming_threads[modality].start()
        return self._streaming_widgets[modality].image

    def stop_streaming(self, modality):
        self._stop_streaming_widget[modality] = True
        self._streaming_threads[modality].join()
        del self._streaming_threads[modality], self._streaming_widgets[modality]

