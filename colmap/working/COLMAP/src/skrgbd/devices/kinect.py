from threading import Event

from pylibfreenect2 import (
    createConsoleLogger, FrameType, Freenect2, LoggerLevel, setGlobalLogger, SyncMultiFrameListener
)
from pylibfreenect2 import CudaPacketPipeline as PacketPipeline
from usb.core import find as finddev

from skrgbd.utils.signal import announce_sensor_freeze
from skrgbd.devices.camera import Camera
from skrgbd.utils.logging import logger
from skrgbd.data.image_utils import pack_float32, equalize_image
from skrgbd.utils.parallel import PropagatingThread as Thread


class Kinect(Camera):
    serial = '025563751347'
    name = 'kinect_v2'

    device = None
    pipeline = None
    rgb_enabled = None
    depth_enabled = None
    ir_enabled = None

    # Setup procedures
    # ----------------
    def __init__(self, rgb_enabled=True, depth_enabled=True, ir_enabled=True):
        super().__init__()
        self.rgb_enabled = rgb_enabled
        self.depth_enabled = depth_enabled
        self.ir_enabled = ir_enabled
        self.got_frame = Event()

        self._init()

    def _init(self):
        logger.debug(f'{self.name}: Reset USB device')
        finddev(idVendor=0x045e, idProduct=0x02c4).reset()

        logger.debug(f'{self.name}: Init freenect')
        self.freenect = Freenect2()
        logger.debug(f'{self.name}: Init pipeline')
        self.pipeline = PacketPipeline()
        logger.debug(f'{self.name}: Init device')
        self.device = self.freenect.openDevice(self.serial.encode(), pipeline=self.pipeline)

        frame_type = FrameType.Color if self.rgb_enabled else 0
        if self.depth_enabled:
            frame_type |= FrameType.Depth
        if self.ir_enabled:
            frame_type |= FrameType.Ir

        self.listener = SyncMultiFrameListener(frame_type)
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)

        # loglevel = {
        #     40: LoggerLevel.Error,
        #     30: LoggerLevel.Warning,
        #     20: LoggerLevel.Info,
        #     10: LoggerLevel.Debug,
        # }[logger.level]
        loglevel = LoggerLevel.Warning
        kinect_logger = createConsoleLogger(loglevel)
        setGlobalLogger(kinect_logger)

    def __del__(self):
        self.device.close()
        del self.listener

    def start(self):
        logger.debug(f'{self.name}: Start')
        self.device.start()
        self.started = True
        logger.debug(f'{self.name}: Start DONE')

    def stop(self):
        logger.debug(f'{self.name}: Stop')
        self.device.stop()
        self.started = False
        logger.debug(f'{self.name}: Stop DONE')

    def restart(self):
        self.device.stop()
        self.device.close()
        del self.listener, self.device, self.pipeline
        self._init()
        self.start()

    def setup(self, light, preset):
        r"""Kinect does not allow to change the camera parameters, so this procedure does nothing."""
        pass

    # Scanning procedures
    # -------------------
    def snap_frame(self, equalize_ir=False, compressed=False, timeout=8):
        r"""Captures data from the device.

        Parameters
        ----------
        equalize_ir : bool
            If True, applies adaptive histogram equalization to IR image.
        compressed : bool
            If True, returns

        Returns
        -------
        frame : dict
            image : np.ndarray or None
                of shape [height, width, 3].
            depth : np.ndarray or None
                of shape [height_d, width_d], float32 in millimeters,
                or compressed to [height_d, width_d, 4] uint8.
            ir : np.ndarray or None
                of shape [height_d, width_d], float32 in [0,1] range,
                or in [0, 65535] range compressed to [height_d, width_d, 4] uint8.
        """
        logger.debug(f'{self.name}: Taking image')
        frames = self.wait_for_new_frame(timeout)

        # we copy frames to release them in pylibfreenect2
        frame = dict()
        if self.rgb_enabled:
            frame['image'] = frames['color'].asarray()[:, ::-1, [2, 1, 0]].copy()

        if self.depth_enabled:
            frame['depth'] = frames['depth'].asarray()[:, ::-1]
            if compressed:
                frame['depth'] = pack_float32(frame['depth'])
            else:
                frame['depth'] = frame['depth'].copy()

        if self.ir_enabled:
            frame['ir'] = frames['ir'].asarray()[:, ::-1]
            if compressed:
                frame['ir'] = pack_float32(frame['ir'])
            else:
                frame['ir'] = frame['ir'] / 65535
                if equalize_ir:
                    frame['ir'] = equalize_image(frame['ir'], 51)

        self.listener.release(frames)
        logger.debug(f'{self.name}: Taking image DONE')
        return frame

    def wait_for_new_frame(self, timeout=8, attempts_n=2):
        for restart_i in range(attempts_n):
            frames = [None]

            def get_frame_thread():
                frames[0] = self.listener.waitForNewFrame()
                self.got_frame.set()

            self.got_frame.clear()
            frame_thread = Thread(target=get_frame_thread)
            frame_thread.start()
            self.got_frame.wait(timeout)
            if not self.got_frame.is_set():
                logger.warning(f'{self.name}: Not responding, trying restart')
                self.restart()
            else:
                if restart_i > 0:
                    logger.warning(f'{self.name}: Restart successful')
                return frames[0]

        announce_sensor_freeze()
        raise RuntimeError(f'{self.name} not responding')

    # Streaming procedures
    # --------------------
    def start_streaming(self, modality, im_slice=None, figsize_base=5, figsize=None, ticks=True, frames=True, name=True):
        if im_slice is None:
            im_slice = slice(None, None, None)
            if figsize is None:
                if modality == 'image':
                    figsize = (figsize_base, figsize_base * 9 / 16)
                else:
                    figsize = (figsize_base, figsize_base * 53 / 64)
        else:
            im = self.snap_frame()[modality][im_slice]
            h, w = im.shape[:2]
            figsize = (figsize_base, figsize_base * h / w)

        if modality.startswith('ir'):
            kwargs = dict(cmap='gray', vmin=0, vmax=1)
        else:
            kwargs = dict()

        get_image = lambda: self.snap_frame(equalize_ir=True)[modality][im_slice]
        return super().start_streaming(modality, get_image, figsize, ticks, frames, name, **kwargs)
