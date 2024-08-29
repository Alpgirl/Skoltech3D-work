from threading import Event, RLock
import pickle
from time import sleep, time

from usb.core import find as finddev
from PIL import Image
import numpy as np
import pyrealsense2 as rs
import torch

from skrgbd.utils.signal import announce_sensor_freeze
from skrgbd.devices.camera import Camera
from skrgbd.utils.logging import logger
from skrgbd.image_processing.auto_exposure import pick_camera_settings
from skrgbd.image_processing.matting import segment_fg

from skrgbd.utils.parallel import ThreadSet, PropagatingThread as Thread


class RealSense(Camera):
    r"""

    Parameters
    ----------
    rgb_enabled : bool
        If False swith off the RGB stream.
    depth_enabled : bool
        If False swith off the depth stream.
    ir_enabled : bool
        If False swith off the IR stream.
    """

    name = 'real_sense'
    profile = None
    device = None
    config = None
    rgb_enabled = None
    depth_enabled = None
    ir_enabled = None

    best_gain = 0
    fast_exposure = 33_3
    best_ir_gain = 16
    fast_ir_exposure = 32_000
    exp_min = 1
    exp_max = 1000_0
    gain_min = 0
    gain_max = 128
    exp_ir_min = 1
    exp_ir_max = 165_000
    gain_ir_min = 16
    gain_ir_max = 248

    # Setup procedures
    # ----------------
    def __init__(self, rgb_enabled=True, depth_enabled=True, ir_enabled=True):
        super().__init__()
        logger.debug(f'{self.name}: Reset USB device')
        finddev(idVendor=0x8086, idProduct=0x0b07).reset()

        logger.debug(f'{self.name}: Setup')
        ctx = rs.context()
        devices = ctx.query_devices()
        assert len(devices) == 1
        device = devices[0]

        sensors = device.query_sensors()
        assert len(sensors) == 2
        depth_camera, rgb_camera = sensors
        if depth_camera.is_color_sensor():
            rgb_camera, depth_camera = depth_camera, rgb_camera
        assert rgb_camera.is_color_sensor() and depth_camera.is_depth_sensor()
        self.rgb_camera = rgb_camera
        self.depth_camera = depth_camera

        # Check the available profiles in realsense-viewer.
        # RealSense performs some temporal filtering to compute the depth map,
        # so the higher framerate results in a faster "adaptation" of the depthmap to the new pose.
        # However, the framerate > 15 results in a severe framedrop at the highest resolution
        # when all the streams are enabled.
        # Also, there are some freezes with framerate 15 when setting the exposure,
        # e.g try with fps=15 to set_rgb_exposure(1600) and then set_ir_exposure(80_000),
        # so we use the lowest fps possible.
        fps = 6
        rgb_width = 1920
        rgb_height = 1080
        depth_width = 1280
        depth_height = 720

        rgb_profile = None
        for _ in rgb_camera.profiles:
            _ = _.as_video_stream_profile()
            if _.fps() == fps and _.format() == rs.format.rgb8 and _.width() == rgb_width and _.height() == rgb_height:
                rgb_profile = _
                break
        assert rgb_profile is not None

        depth_profiles = []
        for _ in depth_camera.profiles:
            _ = _.as_video_stream_profile()

            correct_fps_res = _.fps() == fps and _.width() == depth_width and _.height() == depth_height
            correct_format = ((_.stream_name().startswith('Infrared') and _.format() == rs.format.y8)
                              or (_.stream_name().startswith('Depth') and _.format() == rs.format.z16))
            if correct_fps_res and correct_format:
                depth_profiles.append(_)
        assert len(depth_profiles) == 3

        self.rgb_enabled = rgb_enabled
        self.depth_enabled = depth_enabled
        self.ir_enabled = ir_enabled

        if rgb_enabled:
            rgb_camera.open(rgb_profile)
        if not ir_enabled:
            depth_profiles = list(filter(lambda p: not p.stream_name().startswith('Infrared'), depth_profiles))
        if not depth_enabled:
            depth_profiles = list(filter(lambda p: not p.stream_name().startswith('Depth'), depth_profiles))
        if self.ir_enabled or self.depth_enabled:
            depth_camera.open(depth_profiles)

        self._min_frame_timestamp = 0
        self.rgb_frames = FrameBuffer()
        self.ir1_frames = FrameBuffer()
        self.ir2_frames = FrameBuffer()
        self.depth_frames = FrameBuffer()

        self.rgb_exposure = -1
        self.rgb_gain = -1
        self.ir_exposure = -1
        self.ir_gain = -1
        self.laser_power = -1
        self.laser_is_on = False

        self.exposure_presets = dict()
        self.gain_presets = dict()
        self.ir_exposure_presets = dict()
        self.ir_gain_presets = dict()
        self.laser_power_presets = dict()
        self._bg_frame = None
        self._fg_masks = {'image': None, 'ir': None}
        self.setup_light_presets()

    def __del__(self):
        if self.started:
            self.stop()
        if self.rgb_enabled:
            self.rgb_camera.close()
        if self.ir_enabled or self.depth_enabled:
            self.depth_camera.close()

    def start(self):
        logger.debug(f'{self.name}: Start streams')
        if self.rgb_enabled:
            self.rgb_camera.start(self._frame_callback)
        if self.depth_enabled or self.ir_enabled:
            self.depth_camera.start(self._frame_callback)
        self.set_initial_sensor_options()
        self.started = True
        logger.debug(f'{self.name}: Start streams DONE')

    def stop(self):
        logger.debug(f'{self.name}: Stop streams')
        if self.rgb_enabled:
            self.rgb_camera.stop()
        if self.ir_enabled or self.depth_enabled:
            self.depth_camera.stop()
        self.started = False
        logger.debug(f'{self.name}: Stop streams DONE')

    def setup(self, light, preset):
        r"""Sets up the cameras for the light source with the specific imaging preset.

        Parameters
        ----------
        light : {'flash', 'ambient_low', 'room_lights', default}
        preset : {'best', 'fast', 'calibration'}
        """
        logger.debug(f'{self.name}: Setup for {preset} at {light}')
        self.set_rgb_exposure(self.exposure_presets[(light, preset)])
        self.set_rgb_gain(self.gain_presets[(light, preset)])
        self.set_ir_exposure(self.ir_exposure_presets[(light, preset)])
        self.set_ir_gain(self.ir_gain_presets[(light, preset)])
        if self.laser_is_on:
            self.set_laser_power(self.laser_power_presets[(light, preset)])
        logger.debug(f'{self.name}: Setup for {preset} at {light} DONE')

    def setup_light_presets(self):
        light, preset = 'room_lights', 'calibration'
        self.exposure_presets[(light, preset)] = 40_0
        self.gain_presets[(light, preset)] = self.best_gain
        self.ir_exposure_presets[(light, preset)] = 8_000
        self.ir_gain_presets[(light, preset)] = self.best_ir_gain
        self.laser_power_presets[(light, preset)] = 0

        preset = 'best'
        for light in [
            'soft_left', 'soft_right', 'soft_top', 'hard_left_bottom_far', 'hard_left_bottom_close',
            'hard_left_top_close', 'hard_right_top_far', 'hard_right_top_close', 'hard_left_top_far',
            'hard_right_bottom_close', 'ambient'
        ]:
            self.exposure_presets[(light, preset)] = 40_0
            self.gain_presets[(light, preset)] = self.best_gain
            self.ir_exposure_presets[(light, preset)] = 20_000
            self.ir_gain_presets[(light, preset)] = self.best_ir_gain
            self.laser_power_presets[(light, preset)] = 110
        light = 'flash'
        self.exposure_presets[(light, preset)] = 375_0
        self.gain_presets[(light, preset)] = self.best_gain
        self.ir_exposure_presets[(light, preset)] = 100_000
        self.ir_gain_presets[(light, preset)] = self.best_ir_gain
        self.laser_power_presets[(light, preset)] = 30

        preset = 'fast'
        for light in ['flash', 'ambient_low']:
            self.exposure_presets[(light, preset)] = self.fast_exposure
            self.gain_presets[(light, preset)] = 128
            self.ir_exposure_presets[(light, preset)] = self.fast_ir_exposure
            self.ir_gain_presets[(light, preset)] = 100
            self.laser_power_presets[(light, preset)] = 25

        light, preset = 'room_lights', 'background'
        self.exposure_presets[(light, preset)] = 450_0
        self.gain_presets[(light, preset)] = self.best_gain
        self.ir_exposure_presets[(light, preset)] = 75_000
        self.ir_gain_presets[(light, preset)] = self.best_ir_gain
        self.laser_power_presets[(light, preset)] = 0

    def save_light_presets(self, file):
        light_presets = dict(exposure_presets=self.exposure_presets, gain_presets=self.gain_presets,
                             ir_exposure_presets=self.ir_exposure_presets, ir_gain_presets=self.ir_gain_presets,
                             laser_power_presets=self.laser_power_presets)
        pickle.dump(light_presets, open(file, 'wb'))

    def load_light_presets(self, file):
        light_presets = pickle.load(open(file, 'rb'))
        self.exposure_presets = light_presets['exposure_presets']
        self.gain_presets = light_presets['gain_presets']
        self.ir_exposure_presets = light_presets['ir_exposure_presets']
        self.ir_gain_presets = light_presets['ir_gain_presets']
        self.laser_power_presets = light_presets['laser_power_presets']

    def copy_light_preset(self, src, dst):
        self.exposure_presets[dst] = self.exposure_presets[src]
        self.gain_presets[dst] = self.gain_presets[src]
        self.ir_exposure_presets[dst] = self.ir_exposure_presets[src]
        self.ir_gain_presets[dst] = self.ir_gain_presets[src]
        self.laser_power_presets[dst] = self.laser_power_presets[src]

    def set_initial_sensor_options(self):
        logger.debug(f'{self.name}: Setup streams')
        # RGB
        self.rgb_camera.set_option(rs.option.enable_auto_exposure, 0)
        self.rgb_camera.set_option(rs.option.gain, 0)
        self.rgb_gain = 0
        self.rgb_camera.set_option(rs.option.exposure, 300)
        self.rgb_exposure = 300

        self.rgb_camera.set_option(rs.option.brightness, 0)
        self.rgb_camera.set_option(rs.option.contrast, 50)
        self.rgb_camera.set_option(rs.option.saturation, 64)
        self.rgb_camera.set_option(rs.option.hue, 0)
        self.rgb_camera.set_option(rs.option.gamma, 300)

        self.rgb_camera.set_option(rs.option.enable_auto_white_balance, False)

        # IR
        self.depth_camera.set_option(rs.option.laser_power, 150)
        self.laser_power = 150
        self.depth_camera.set_option(rs.option.emitter_enabled, False)
        self.laser_is_on = False
        self.depth_camera.set_option(rs.option.enable_auto_exposure, False)
        self.depth_camera.set_option(rs.option.gain, 16)
        self.ir_gain = 16
        self.depth_camera.set_option(rs.option.exposure, 8000)
        self.ir_exposure = 8000

    def set_rgb_exposure(self, exposure):
        r"""
        Parameters
        ----------
        exposure : int
            in range [1, 1000_0] in tenths of ms.
        """
        exposure = int(np.clip(exposure, self.exp_min, self.exp_max))
        if exposure == self.rgb_exposure:
            return
        logger.debug(f'{self.name}: Set RGB exposure to {exposure}')
        self.rgb_exposure = exposure
        safe_set_option(self.rgb_camera, rs.option.exposure, exposure)

    def set_rgb_gain(self, gain):
        r"""
        Parameters
        ----------
        gain : int
            in range [0, 128].
        """
        gain = int(np.clip(gain, self.gain_min, self.gain_max))
        if gain == self.rgb_gain:
            return
        logger.debug(f'{self.name}: Set RGB gain to {gain}')
        self.rgb_gain = gain
        safe_set_option(self.rgb_camera, rs.option.gain, gain)

    def set_ir_exposure(self, exposure):
        r"""
        Parameters
        ----------
        exposure : int
            in range [1, 165_000] in microseconds.
        """
        exposure = int(np.clip(exposure, self.exp_ir_min, self.exp_ir_max))
        if exposure == self.ir_exposure:
            return
        logger.debug(f'{self.name}: Set IR exposure to {exposure}')
        self.ir_exposure = exposure
        safe_set_option(self.depth_camera, rs.option.exposure, exposure)

    def set_ir_gain(self, gain):
        r"""
        Parameters
        ----------
        gain : int
            in range [16, 248].
        """
        gain = int(np.clip(gain, self.gain_ir_min, self.gain_ir_max))
        if gain == self.ir_gain:
            return
        logger.debug(f'{self.name}: Set IR gain to {gain}')
        self.ir_gain = gain
        safe_set_option(self.depth_camera, rs.option.gain, gain)

    def ir_auto_exposure_on(self):
        safe_set_option(self.depth_camera, rs.option.enable_auto_exposure, True)

    def ir_auto_exposure_off(self):
        safe_set_option(self.depth_camera, rs.option.enable_auto_exposure, False)

    def laser_on(self):
        if self.laser_is_on:
            return
        self.laser_is_on = True
        safe_set_option(self.depth_camera, rs.option.emitter_enabled, True)

    def laser_off(self):
        if not self.laser_is_on:
            return
        self.laser_is_on = False
        safe_set_option(self.depth_camera, rs.option.emitter_enabled, False)

    def set_laser_power(self, laser_power):
        r"""
        Parameters
        ----------
        laser_power : int
            in range [0, 360] with the step 30.
        """
        laser_power = np.clip(round(laser_power / 30) * 30, 0, 360)
        if laser_power == self.laser_power:
            return
        logger.debug(f'{self.name}: Set laser power to {laser_power}')
        self.laser_power = laser_power
        safe_set_option(self.depth_camera, rs.option.laser_power, laser_power)

    def auto_white_balance(self):
        logger.debug(f'{self.name}: Auto white balance')
        self.auto_exposure_on()
        safe_set_option(self.rgb_camera, rs.option.enable_auto_white_balance, True)
        sleep(8)
        safe_set_option(self.rgb_camera, rs.option.enable_auto_white_balance, False)
        self.auto_exposure_off()
        logger.debug(f'{self.name}: Auto white balance DONE')

    def auto_exposure_on(self):
        safe_set_option(self.rgb_camera, rs.option.enable_auto_exposure, True)
        self.rgb_exposure = -1

    def auto_exposure_off(self):
        safe_set_option(self.rgb_camera, rs.option.enable_auto_exposure, False)

    def pick_settings(self, light, preset, ir_frames_n=1):
        def rgb_target():
            self._pick_rgb_settings(light, preset)

        def ir_target():
            self._pick_ir_settings(light, preset, ir_frames_n)
            self._pick_laser_settings(light, preset, ir_frames_n)

        ThreadSet([Thread(target=rgb_target), Thread(target=ir_target)]).start_and_join()

    def _pick_rgb_settings(self, light, preset):
        logger.debug(f'{self.name}: Pick settings for RGB at {light} at {preset}')
        exp_low, exp_high = self.exp_min, self.exp_max
        gain_low, gain_high = self.best_gain, self.gain_max

        i, j = self._fg_masks['image'].nonzero()
        bbox_i = slice(i.min(), i.max())
        bbox_j = slice(j.min(), j.max())
        fg_mask = torch.from_numpy(self._fg_masks['image'][bbox_i, bbox_j])

        def get_img():
            self.rgb_frames.clear(1)
            self.rgb_frames.wait_full()
            img = self.rgb_frames[0]
            img = torch.from_numpy(img)
            return img[bbox_i, bbox_j]

        exposure, gain = pick_camera_settings(
            preset, get_img, fg_mask, self.set_rgb_exposure, exp_low, exp_high, self.fast_exposure,
            self.set_rgb_gain, gain_low, gain_high, self.best_gain)

        self.exposure_presets[(light, preset)] = exposure
        self.gain_presets[(light, preset)] = gain
        logger.debug(f'{self.name}: Pick settings for RGB at {light} at {preset} DONE exp={exposure}, gain={gain}')

    def _pick_ir_settings(self, light, preset, ir_frames_n):
        logger.debug(f'{self.name}: Pick settings for IR at {light} at {preset}')
        exp_low, exp_high = self.exp_ir_min, self.exp_ir_max
        gain_low, gain_high = self.best_ir_gain, self.gain_ir_max

        i, j = self._fg_masks['ir'].nonzero()
        bbox_i = slice(i.min(), i.max())
        bbox_j = slice(j.min(), j.max())
        fg_mask = torch.from_numpy(self._fg_masks['ir'][bbox_i, bbox_j])

        def get_img():
            self.ir1_frames.clear(ir_frames_n)
            self.ir1_frames.wait_full()
            img = np.mean([f[bbox_i, bbox_j] for f in self.ir1_frames.frames], 0)
            img = torch.from_numpy(img).round().clamp(0, 255).byte()
            return img

        self.laser_on()
        self.set_laser_power(30)
        exposure, gain = pick_camera_settings(
            preset, get_img, fg_mask, self.set_ir_exposure, exp_low, exp_high, self.fast_ir_exposure,
            self.set_ir_gain, gain_low, gain_high, self.best_ir_gain)
        self.ir_exposure_presets[(light, preset)] = exposure
        self.ir_gain_presets[(light, preset)] = gain
        logger.debug(f'{self.name}: Pick settings for IR at {light} at {preset} DONE exp={exposure}, gain={gain}')

    def _pick_laser_settings(self, light, preset, ir_frames_n):
        logger.debug(f'{self.name}: Pick laser settings at {light} at {preset}')
        self.laser_on()
        self.set_ir_gain(self.ir_gain_presets[(light, preset)])
        self.set_ir_exposure(self.ir_exposure_presets[(light, preset)])

        best_valid_pixels_n = 0
        best_laser_power = 30
        for laser_power in range(30, 360 + 1, 30):
            self.set_laser_power(laser_power)
            self.depth_frames.clear(max(12, ir_frames_n))
            self.depth_frames.wait_full()
            valid_pixels_n = np.min([
                np.count_nonzero(depth[self._fg_masks['ir']]) for depth in self.depth_frames.frames])
            logger.debug(f'{self.name}: Valid pixels {valid_pixels_n} at {laser_power}')
            if valid_pixels_n > best_valid_pixels_n:
                best_valid_pixels_n = valid_pixels_n
                best_laser_power = laser_power

        self.laser_power_presets[(light, preset)] = best_laser_power
        logger.debug(f'{self.name}: Pick laser settings at {light} at {preset} DONE power={best_laser_power}')

    def set_bg_frame(self):
        self.setup('room_lights', 'background')
        self._bg_frame = self.snap_frame(compressed=True)

    def set_fg_masks(self, method='nn', threshold=25):
        self.setup('room_lights', 'background')
        fg_frame = self.snap_frame(compressed=True)

        self._fg_masks = dict()
        if method == 'classic':
            _ = fg_frame['image'].astype(np.int16) - self._bg_frame['image'].astype(np.int16)
            self._fg_masks['image'] = (np.abs(_) > threshold).any(-1)

            _ = fg_frame['ir'].astype(np.int16) - self._bg_frame['ir'].astype(np.int16)
            self._fg_masks['ir'] = np.abs(_) > threshold
        elif method == 'nn':
            img = fg_frame['image']
            bg = self._bg_frame['image']
            img = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
            bg = torch.from_numpy(bg).permute(2, 0, 1).float().div(255)
            self._fg_masks['image'] = segment_fg(img, bg).numpy()

            img = fg_frame['ir']
            bg = self._bg_frame['ir']
            img = torch.from_numpy(img).float().div(255).unsqueeze(0).expand(3, -1, -1)
            bg = torch.from_numpy(bg).float().div(255).unsqueeze(0).expand(3, -1, -1)
            self._fg_masks['ir'] = segment_fg(img, bg).numpy()
        elif method == 'full':
            img = fg_frame['image']
            self._fg_masks['image'] = np.ones_like(img[..., 0], dtype=bool)
            img = fg_frame['ir']
            self._fg_masks['ir'] = np.ones_like(img, dtype=bool)
        else:
            raise ValueError(f'Unknown method {method}')

    # Scanning procedures
    # -------------------
    def snap_frame(self, equalize_ir=False, compressed=False, timeout=8):
        r"""Captures data from the device.

        Parameters
        ----------
        compressed : bool
            If True, returns ir in the raw RealSense uint8 format.

        Returns
        -------
        frame : dict
            image : np.ndarray
                of shape [height, width, 3].
            depth : np.ndarray
                of shape [height_d, width_d], uint16 in millimeters.
            ir : np.ndarray
                of shape [height_d, width_d] in [0,1] range, or in uint8 if `compressed` is True.
            ir_left : np.ndarray
                of shape [height_d, width_d] in [0,1] range, or in uint8 if `compressed` is True.
        """
        self._min_frame_timestamp = time() * 1000
        logger.debug(f'{self.name}: Taking image')
        if self.rgb_enabled:
            self.rgb_frames.clear(1)
        if self.depth_enabled:
            self.depth_frames.clear(1)
        if self.ir_enabled:
            self.ir1_frames.clear(1)
            self.ir2_frames.clear(1)

        frame = dict()
        if self.rgb_enabled:
            self.rgb_frames.wait_full(timeout)
            frame['image'] = self.rgb_frames[0]

        if self.depth_enabled:
            self.depth_frames.wait_full(timeout)
            frame['depth'] = self.depth_frames[0]

        if self.ir_enabled:
            for label, ir_frames in (['ir', self.ir1_frames], ['ir_right', self.ir2_frames]):
                ir_frames.wait_full(timeout)
                image = ir_frames[0]
                if not compressed:
                    image = image / 255
                frame[label] = image

        logger.debug(f'{self.name}: Taking image DONE')
        return frame

    # IO procedures
    # -------------
    def save_ir_npy(self, path, frame):
        for side, image, side_path in [['left', frame['ir'], f'{path}.npy'],
                                       ['right', frame['ir_right'], f'{path}r.npy']]:
            logger.debug(f'{self.name}: Saving IR {side} to {side_path}')
            np.save(side_path, image)

    def save_ir_png(self, path, frame):
        for side, image, side_path in [['left', frame['ir'], f'{path}.png'],
                                       ['right', frame['ir_right'], f'{path}r.png']]:
            logger.debug(f'{self.name}: Saving IR {side} to {side_path}')
            Image.fromarray(image).save(side_path)

    def save_ir_calib(self, root_dir, image_name, frame):
        for side, ir in [['left', 'ir'], ['right', 'ir_right']]:
            path = root_dir / f'{self.name}_{ir}'
            path.mkdir(parents=True, exist_ok=True)
            path = path / f'{image_name}.png'

            image = frame[ir]
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            logger.debug(f'{self.name}: Saving IR {side} to {path}')
            Image.fromarray(image).save(path)

    # Streaming procedures
    # --------------------
    def start_streaming(self, modality, im_slice=None, figsize_base=5, figsize=None, ticks=True, frames=True, name=True):
        self._min_frame_timestamp = 0
        if modality == 'image':
            def get_image():
                self.rgb_frames.clear(1)
                self.rgb_frames.wait_full()
                return self.rgb_frames[0][im_slice]
        elif modality == 'ir':
            def get_image():
                self.ir1_frames.clear(1)
                self.ir1_frames.wait_full()
                return self.ir1_frames[0][im_slice] / 255
        elif modality == 'ir_right':
            def get_image():
                self.ir2_frames.clear(1)
                self.ir2_frames.wait_full()
                return self._ir2_frames[0][im_slice] / 255
        else:
            raise ValueError

        if im_slice is None:
            im_slice = slice(None, None, None)
            if figsize is None:
                figsize = (figsize_base, figsize_base * 9 / 16)
        else:
            im = get_image()
            h, w = im.shape[:2]
            figsize = (figsize_base, figsize_base * h / w)

        if modality.startswith('ir'):
            kwargs = dict(cmap='gray', vmin=0, vmax=1)
        else:
            kwargs = dict()

        return super().start_streaming(modality, get_image, figsize, ticks, frames, name, **kwargs)

    # Misc
    # ----
    def _frame_callback(self, frame):
        if frame.timestamp < self._min_frame_timestamp:
            return
        stream = frame.get_profile().stream_name()
        if (stream == 'Color') and (not self.rgb_frames.is_full()) and self._good_rgb_frame(frame):
            self.rgb_frames.add(np.asanyarray(frame.get_data()).copy())
        elif stream == 'Infrared 1' and (not self.ir1_frames.is_full()) and self._good_depth_frame(frame):
            self.ir1_frames.add(np.asanyarray(frame.get_data()).copy())
        elif stream == 'Infrared 2' and (not self.ir2_frames.is_full()) and self._good_depth_frame(frame):
            self.ir2_frames.add(np.asanyarray(frame.get_data()).copy())
        elif stream == 'Depth' and (not self.depth_frames.is_full()) and self._good_depth_frame(frame):
            self.depth_frames.add(np.asanyarray(frame.get_data()).copy())

    def _good_rgb_frame(self, frame):
        return ((frame.get_frame_metadata(rs.frame_metadata_value.actual_exposure) == self.rgb_exposure)
                and (frame.get_frame_metadata(rs.frame_metadata_value.gain_level) == self.rgb_gain))

    def _good_depth_frame(self, frame):
        r"""
        Notes
        ----- 
        Some first depth frames, which are captured right after internal restart of the stream, are usually very bad,
        apparently because realsense needs some frame statistics for temporal filtering.
        """
        return ((frame.frame_number > 1)
                and (frame.get_frame_metadata(rs.frame_metadata_value.gain_level) == self.ir_gain)
                and (frame.get_frame_metadata(rs.frame_metadata_value.actual_exposure) == self.ir_exposure)
                and (frame.get_frame_metadata(rs.frame_metadata_value.frame_laser_power_mode) == self.laser_is_on)
                and ((not self.laser_is_on)
                     or (frame.get_frame_metadata(rs.frame_metadata_value.frame_laser_power) == self.laser_power)))


class FrameBuffer:
    def __init__(self):
        self.frames = []
        self.size = 0
        self.full_event = Event()
        self.lock = RLock()

    def clear(self, size=None):
        self.lock.acquire()
        if size is not None:
            self.size = size
        self.frames = []
        self.full_event.clear()
        self.lock.release()

    def is_full(self):
        self.lock.acquire()
        is_full = len(self.frames) >= self.size
        self.lock.release()
        return is_full

    def add(self, frame):
        self.lock.acquire()
        self.frames.append(frame)
        if self.is_full():
            self.full_event.set()
        self.lock.release()

    def __getitem__(self, i):
        return self.frames[i]

    def wait_full(self, timeout=None):
        self.full_event.wait(timeout)
        if not self.full_event.is_set():
            announce_sensor_freeze()
            raise RuntimeError(f'{RealSense.name} not responding')


_busy_msg = 'xioctl(VIDIOC_S_CTRL) failed Last Error: Device or resource busy'


def safe_set_option(sensor, option, value):
    while True:
        try:
            return sensor.set_option(option, value)
        except RuntimeError as exception:
            if (len(exception.args) == 1) and (exception.args[0] == _busy_msg):
                logger.debug(f'{RealSense.name}: {_busy_msg}')
                sleep(.1)
