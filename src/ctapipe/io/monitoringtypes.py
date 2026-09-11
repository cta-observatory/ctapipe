"""Types of monitoring data."""

from enum import Enum

__all__ = [
    "TelescopeMonitoringType",
    "MonitoringType",
]


class TelescopeMonitoringType(Enum):
    """Enum of the different telescope-wise Monitoring Types"""

    #: Camera pixel statistics
    PIXEL_STATISTICS = "camera/pixel_statistics"
    #: Camera calibration coefficients
    CAMERA_COEFFICIENTS = "camera/coefficients"
    #: Telescope pointing information
    TELESCOPE_POINTINGS = "pointing"


class MonitoringType(Enum):
    """Enum of the different (sub)array-wide Monitoring Types"""

    #: Weather station data
    WEATHER = "site/weather"
