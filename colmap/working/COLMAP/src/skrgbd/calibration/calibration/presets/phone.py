from skrgbd.calibration.calibration.presets.tag0 import Tag0Calibrator
from skrgbd.calibration.calibration.presets.tag20 import Tag20Calibrator


class PhoneRGBCalibrator0(Tag0Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=109,
            half_window_size=90,
            **kwargs
        )


class PhoneRGBCalibrator20(Tag20Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=109,
            half_window_size=100,
            **kwargs
        )


class PhoneIRCalibrator20(Tag20Calibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            cell_size=4,
            half_window_size=9,
            **kwargs
        )
