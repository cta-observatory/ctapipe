from enum import Enum


class MonitoringType(Enum):
    """Enum of the different Monitoring Types"""

    #: Camera pixel statistics
    PIXEL_STATISTICS = "camera/pixel_statistics"
    #: Camera calibration coefficients
    CAMERA_COEFFICIENTS = "camera/coefficients"
    #: Telescope pointing information
    TELESCOPE_POINTINGS = "pointing"


# Telescope-specific monitoring types (require tel_id parameter)
TELESCOPE_SPECIFIC_MONITORING = {
    MonitoringType.PIXEL_STATISTICS,
    MonitoringType.CAMERA_COEFFICIENTS,
    MonitoringType.TELESCOPE_POINTINGS,
}
