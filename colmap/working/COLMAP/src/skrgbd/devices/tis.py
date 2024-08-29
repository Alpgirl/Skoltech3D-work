from abc import ABC, abstractmethod
import pickle
from time import sleep

import gi
gi.require_version('Tcam', '0.1')
gi.require_version('Gst', '1.0')
from gi.repository import Gst, Tcam
import numpy as np
import torch

from skrgbd.utils.signal import announce_sensor_freeze
from skrgbd.devices.camera import Camera
from skrgbd.utils.parallel import EventVar
from skrgbd.utils.logging import logger
from skrgbd.image_processing.auto_exposure import pick_camera_settings
from skrgbd.image_processing.matting import segment_fg


class TisCamera(Camera, ABC):
    _snap_with_trigger = True  # disable if you need auto exposure/gain
    best_gain = 20
    fast_exposure = 33_333

    exp_min = 20
    exp_max = 4_000_000
    gain_min = 0
    gain_max = 480

    @property
    @abstractmethod
    def serial(self): ...

    @property
    @abstractmethod
    def name(self): ...

    # Setup procedures
    # ----------------
    def __init__(self):
        super().__init__()
        logger.debug(f'{self.name}: Init Gstreamer')
        Gst.init([])
        Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
        logger.debug(f'{self.name}: Launch pipeline')
        self.pipeline = Gst.parse_launch(
            f'tcambin serial={self.serial} name=source ! capsfilter name=caps ! appsink name=sink')
        # alternative pipeline for debugging
        # self.pipeline = Gst.parse_launch(
        #     f'tcambin serial={self.serial} name=source ! capsfilter name=caps ! videoconvert ! ximagesink')
        if not self.pipeline:
            raise RuntimeError

        self.source = self.pipeline.get_by_name('source')
        self.sink = self.pipeline.get_by_name('sink')
        self.caps = self.pipeline.get_by_name('caps')

        self._set_image_format()
        self._set_image_settings()
        # disable frame buffering
        self.sink.set_property('max-buffers', 1)
        self.sink.set_property('drop', True)
        # call self._on_new_image on new frame
        self.sink.set_property('emit-signals', True)
        self.sink.connect('new-sample', self._on_new_image)

        self._image = EventVar()
        self.exposure = -1
        self.gain = -1

        self.exposure_presets = dict()
        self.gain_presets = dict()
        self._bg_frame = None
        self._fg_masks = None
        self.setup_light_presets()

    def __del__(self):
        self.stop()

    def start(self):
        logger.debug(f'{self.name}: Start')
        self.pipeline.set_state(Gst.State.PLAYING)
        sleep(1)
        self.source.set_tcam_property('Whitebalance Auto', False)
        if self._snap_with_trigger:
            self._trigger_mode_on()
        else:
            self._trigger_mode_off()
        logger.debug(f'{self.name}: Start DONE')

    def stop(self):
        logger.debug(f'{self.name}: Stop')
        self.pipeline.set_state(Gst.State.NULL)
        logger.debug(f'{self.name}: Stop DONE')

    def setup(self, light, preset):
        r"""Sets up the camera for the light source with the specific imaging preset.

        Parameters
        ----------
        light : {'flash', 'ambient_low', 'room_lights', default}
        preset : {'best', 'fast', 'calibration'}
        """
        logger.debug(f'{self.name}: Setup for {preset} at {light}')
        self.set_exposure(self.exposure_presets[(light, preset)])
        self.set_gain(self.gain_presets[(light, preset)])
        logger.debug(f'{self.name}: Setup for {preset} at {light} DONE')

    def setup_light_presets(self):
        light, preset = 'room_lights', 'calibration'
        self.exposure_presets[(light, preset)] = 150_000
        self.gain_presets[(light, preset)] = self.best_gain

        preset = 'best'
        for light in [
            'soft_left', 'soft_right', 'soft_top', 'hard_left_bottom_far', 'hard_left_bottom_close',
            'hard_left_top_close', 'hard_right_top_far', 'hard_right_top_close', 'hard_left_top_far',
            'hard_right_bottom_close', 'ambient'
        ]:
            self.exposure_presets[(light, preset)] = 250_000
            self.gain_presets[(light, preset)] = self.best_gain
        light = 'flash'
        self.exposure_presets[(light, preset)] = 2_500_000
        self.gain_presets[(light, preset)] = self.best_gain

        preset = 'fast'
        light = 'flash'
        self.exposure_presets[(light, preset)] = self.fast_exposure
        self.gain_presets[(light, preset)] = 400
        light = 'ambient_low'
        self.exposure_presets[(light, preset)] = self.fast_exposure
        self.gain_presets[(light, preset)] = 393

        light, preset = 'room_lights', 'background'
        # self.exposure_presets[(light, preset)] = 3_000_000  # for classical fg picking
        self.exposure_presets[(light, preset)] = 1_500_000  # for deep fg picking
        self.gain_presets[(light, preset)] = self.best_gain

    def save_light_presets(self, file):
        light_presets = dict(exposure_presets=self.exposure_presets, gain_presets=self.gain_presets)
        pickle.dump(light_presets, open(file, 'wb'))

    def load_light_presets(self, file):
        light_presets = pickle.load(open(file, 'rb'))
        self.exposure_presets = light_presets['exposure_presets']
        self.gain_presets = light_presets['gain_presets']

    def copy_light_preset(self, src, dst):
        self.exposure_presets[dst] = self.exposure_presets[src]
        self.gain_presets[dst] = self.gain_presets[src]

    def set_exposure(self, exposure):
        r"""
        Parameters
        ----------
        exposure : int
            Exposure in us in range 20-4_000_000.
        """
        exposure = int(np.clip(exposure, self.exp_min, self.exp_max))
        if exposure == self.exposure:
            return
        logger.debug(f'{self.name}: Set exposure to {exposure}')
        if self._snap_with_trigger:
            self.source.set_tcam_property('Trigger Mode', False)
        self.exposure = exposure
        self.source.set_tcam_property('Exposure Time (us)', exposure)
        if self._snap_with_trigger:
            self.source.set_tcam_property('Trigger Mode', True)

    def set_gain(self, gain):
        r"""
        Parameters
        ----------
        gain : int
            Gain in range 0-480. Values near 0 lead to pink star artefacts.
        """
        gain = int(np.clip(gain, self.gain_min, self.gain_max))
        if gain == self.gain:
            return
        logger.debug(f'{self.name}: Set gain to {gain}')
        if self._snap_with_trigger:
            self.source.set_tcam_property('Trigger Mode', False)
        self.gain = gain
        self.source.set_tcam_property('Gain', gain)
        if self._snap_with_trigger:
            self.source.set_tcam_property('Trigger Mode', True)

    def auto_white_balance(self):
        logger.debug(f'{self.name}: Auto white balance')
        self.auto_gain_on()
        self.auto_exposure_on()
        self.source.set_tcam_property('Whitebalance Auto', True)
        if self._snap_with_trigger:
            self._trigger_mode_off()
            sleep(5)
            self._trigger_mode_on()
        else:
            sleep(5)
        self.source.set_tcam_property('Whitebalance Auto', False)
        self.auto_exposure_off()
        self.auto_gain_off()
        logger.debug(f'{self.name}: Auto white balance DONE')

    def auto_gain_on(self):
        self.source.set_tcam_property('Gain Auto', True)
        self.gain = -1

    def auto_gain_off(self):
        self.source.set_tcam_property('Gain Auto', False)

    def auto_exposure_on(self):
        self.source.set_tcam_property('Exposure Auto', True)
        self.exposure = -1

    def auto_exposure_off(self):
        self.source.set_tcam_property('Exposure Auto', False)

    def pick_settings(self, light, preset):
        logger.debug(f'{self.name}: Pick settings for RGB at {light} at {preset}')
        exp_low, exp_high = self.exp_min, self.exp_max
        gain_low, gain_high = self.best_gain, self.gain_max

        i, j = self._fg_masks.nonzero()
        bbox_i = slice(i.min(), i.max())
        bbox_j = slice(j.min(), j.max())
        fg_mask = torch.from_numpy(self._fg_masks[bbox_i, bbox_j])

        def get_img():
            img = self.snap_frame()['image']
            img = torch.from_numpy(img)
            return img[bbox_i, bbox_j]

        exposure, gain = pick_camera_settings(
            preset, get_img, fg_mask, self.set_exposure, exp_low, exp_high, self.fast_exposure,
            self.set_gain, gain_low, gain_high, self.best_gain)
        self.exposure_presets[(light, preset)] = exposure
        self.gain_presets[(light, preset)] = gain
        logger.debug(f'{self.name}: Pick settings for RGB at {light} at {preset} DONE exp={exposure}, gain={gain}')

    def set_bg_frame(self):
        self.setup('room_lights', 'background')
        self.snap_frame()  # blank snap to ensure the setting is applied
        self._bg_frame = self.snap_frame()['image']

    def set_fg_masks(self, method='nn', threshold=25):
        self.setup('room_lights', 'background')
        self.snap_frame()  # blank snap to ensure the setting is applied
        fg_frame = self.snap_frame()['image']
        if method == 'classic':
            self._fg_masks = (np.abs(fg_frame.astype(np.int16) - self._bg_frame.astype(np.int16)) > threshold).any(-1)
        elif method == 'nn':
            fg_frame = torch.from_numpy(fg_frame).permute(2, 0, 1).float().div(255)
            bg_frame = torch.from_numpy(self._bg_frame).permute(2, 0, 1).float().div(255)
            self._fg_masks = segment_fg(fg_frame, bg_frame).numpy()
        elif method == 'full':
            self._fg_masks = np.ones_like(fg_frame[..., 0], dtype=bool)
        else:
            raise ValueError(f'Unknown method {method}')

    # Scanning procedures
    # -------------------
    def snap_frame(self, equalize_ir=False, compressed=False, timeout=8):
        r"""
        Returns
        -------
        frame : dict
            image : np.ndarray
                of shape [height, width, 3]
        """
        logger.debug(f'{self.name}: Taking image')
        self._image.clear()
        if self._snap_with_trigger:
            success = self.source.set_tcam_property('Software Trigger', True)
            if not success:
                raise RuntimeError
        image = self._image.wait(timeout=timeout)
        if not self._image.is_set():
            announce_sensor_freeze()
            raise RuntimeError(f'{self.name} not responding')
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
                figsize = (figsize_base, figsize_base * 128 / 153)
        else:
            im = self.snap_frame()[modality][im_slice]
            h, w = im.shape[:2]
            figsize = (figsize_base, figsize_base * h / w)

        kwargs = dict()
        get_image = lambda: self.snap_frame()[modality][im_slice]
        return super().start_streaming(modality, get_image, figsize, ticks, frames, name, **kwargs)

    # Misc
    # ----
    def _set_image_format(self):
        r"""Sets the framerate and the image height, width, and color format."""
        logger.debug(f'{self.name}: Setting image format')
        caps = Gst.Caps.new_empty()
        structure = Gst.Structure.new_from_string('video/x-raw')
        structure.set_value('format', 'RGB')
        structure.set_value('width', 2448)
        structure.set_value('height', 2048)
        structure.set_value('framerate', Gst.Fraction(15, 1))
        caps.append_structure(structure)
        structure.free()
        self.caps.set_property('caps', caps)
        logger.debug(f'{self.name}: Setting image format DONE')

    def _set_image_settings(self):
        logger.debug(f'{self.name}: Setting image settings')

        self.source.set_tcam_property('Exposure Auto', False)
        self.source.set_tcam_property('Gain Auto', False)
        self.source.set_tcam_property('Brightness', 0)
        # exposure / exp(gain / 87.97) == const
        # we use a slightly nonzero gain to remove purple dot noise in overexposed areas
        self.source.set_tcam_property('Gain', 20)
        self.source.set_tcam_property('Exposure Time (us)', 200_000)

        self.source.set_tcam_property('Saturation', 64)
        self.source.set_tcam_property('Hue', 0)

        logger.debug(f'{self.name}: Setting image settings DONE')

    def _trigger_mode_on(self):
        self._snap_with_trigger = True
        self.source.set_tcam_property('Trigger Mode', True)
        sleep(1)

    def _trigger_mode_off(self):
        self._snap_with_trigger = False
        self.source.set_tcam_property('Trigger Mode', False)
        sleep(1)

    def _on_new_image(self, appsink):
        r"""Callback for asynchronous image capture."""
        sample = appsink.emit('pull-sample')
        if not sample:
            raise RuntimeError
        if (not self._snap_with_trigger) and self._image.is_set():
            return Gst.FlowReturn.OK

        # get the dimensions
        caps = sample.get_caps()
        height = caps.get_structure(0).get_value('height')
        width = caps.get_structure(0).get_value('width')

        # get the actual data
        buffer = sample.get_buffer()
        # get read access to the buffer data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError

        self._image.set(np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3))

        # Clean up the buffer mapping
        buffer.unmap(map_info)

        return Gst.FlowReturn.OK


class RightTisCamera(TisCamera):
    serial = '36710103'
    name = 'tis_right'


class LeftTisCamera(TisCamera):
    serial = '06710488'
    name = 'tis_left'
