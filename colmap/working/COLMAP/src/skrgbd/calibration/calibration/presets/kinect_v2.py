from skrgbd.calibration.calibration.presets.tag0 import Tag0Calibrator
from skrgbd.calibration.calibration.presets.tag20 import Tag20Calibrator


class KinectV2RGBCalibrator(Tag0Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=21,
            half_window_size=16,
            **kwargs
        )


class KinectV2IRCalibrator(Tag0Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=8,
            half_window_size=6,
            **kwargs
        )


class KinectV2IRCalibrator20(Tag20Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=8,
            half_window_size=6,
            **kwargs
        )
