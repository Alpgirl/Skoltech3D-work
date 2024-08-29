from skrgbd.calibration.calibration.presets.tag0 import Tag0Calibrator
from skrgbd.calibration.calibration.presets.tag20 import Tag20Calibrator


class RealSenseRGBCalibrator(Tag0Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=21,
            half_window_size=20,
            **kwargs
        )


class RealSenseIRCalibrator(Tag0Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=14,
            half_window_size=10,
            **kwargs
        )


class RealSenseIRCalibrator20(Tag20Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=14,
            half_window_size=10,
            **kwargs
        )
