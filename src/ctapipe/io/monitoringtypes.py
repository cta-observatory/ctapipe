from enum import Enum, auto


class MonitoringTypes(Enum):
    """Enum of the different Monitoring Types"""

    #: Raw data in common format, with preliminary calibration
    PIXEL_STATISTICS = auto()
    #: Camera calibration coefficients
    CAMERA_COEFFICIENTS = auto()
    #: raw archived data in common format, with optional zero suppression
    POINTING = auto()
    
