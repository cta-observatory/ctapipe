from enum import Enum


class MonitoringType(Enum):
    """Enum of the different Monitoring Types"""

    #: Camera pixel statistics
    PIXEL_STATISTICS = "camera/pixel_statistics"
    #: Camera calibration coefficients
    CAMERA_COEFFICIENTS = "camera/coefficients"
    #: Telescope pointing information
    TELESCOPE_POINTINGS = "pointing"
