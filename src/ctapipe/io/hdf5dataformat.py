""" HDF5 Data Format Constants """

__all__ = [
    "R0_GROUP",
    "R1_GROUP",
    "DL0_GROUP",
    "DL1_GROUP",
    "DL2_GROUP",
    "SIMULATION_GROUP",
    "CONFIG_GROUP",
    "SCHEDULING_BLOCK_GROUP",
    "OBSERVATION_BLOCK_GROUP",
    "SIMULATION_RUN_GROUP",
    "CONFIG_TEL_POINTING_GROUP",
    "ATMOSPHERE_DENSITY_PROFILE_GROUP",
    "DL1_IMAGE_STATISTICS_GROUP",
    "DL2_EVENT_STATISTICS_GROUP",
    "SHOWER_DISTRIBUTION_GROUP",
    "SIMULATION_SHOWER_GROUP",
    "SIMULATION_IMPACT_GROUP",
    "SIMULATION_IMAGES_GROUP",
    "SIMULATION_PARAMETERS_GROUP",
    "R0_TEL_GROUP",
    "R1_TEL_GROUP",
    "DL1_SUBARRAY_GROUP",
    "DL1_SUBARRAY_TRIGGER_GROUP",
    "DL1_TEL_GROUP",
    "DL1_TEL_TRIGGER_GROUP",
    "DL1_TEL_IMAGES_GROUP",
    "DL1_TEL_PARAMETERS_GROUP",
    "DL1_TEL_MUON_GROUP",
    "DL2_TEL_GROUP",
    "DL2_SUBARRAY_GROUP",
    "DL0_TEL_POINTING_GROUP",
    "DL1_SUBARRAY_POINTING_GROUP",
    "DL1_TEL_POINTING_GROUP",
    "DL1_TEL_CALIBRATION_GROUP",
    "DL1_CAMERA_MONITORING_GROUP",
    "DL1_CAMERA_COEFFICIENTS_GROUP",
    "DL1_PIXEL_STATISTICS_GROUP",
    "DL1_SKY_PEDESTAL_IMAGE_GROUP",
    "DL1_FLATFIELD_IMAGE_GROUP",
    "DL1_FLATFIELD_PEAK_TIME_GROUP",
    "DL1_COLUMN_NAMES",
]

# Configuration, service, and simulation group
CONFIG_GROUP = "/configuration"
SCHEDULING_BLOCK_GROUP = "/configuration/observation/scheduling_block"
OBSERVATION_BLOCK_GROUP = "/configuration/observation/observation_block"
SIMULATION_RUN_GROUP = "/configuration/simulation/run"
CONFIG_TEL_POINTING_GROUP = "/configuration/telescope/pointing"
DL1_IMAGE_STATISTICS_GROUP = "/dl1/service/image_statistics"
DL2_EVENT_STATISTICS_GROUP = "/dl2/service/tel_event_statistics"
SIMULATION_GROUP = "/simulation"
ATMOSPHERE_DENSITY_PROFILE_GROUP = "/simulation/service/atmosphere_density_profile"
SHOWER_DISTRIBUTION_GROUP = "/simulation/service/shower_distribution"
SIMULATION_SHOWER_GROUP = "/simulation/event/subarray/shower"
SIMULATION_IMPACT_GROUP = "/simulation/event/telescope/impact"
SIMULATION_IMAGES_GROUP = "/simulation/event/telescope/images"
SIMULATION_PARAMETERS_GROUP = "/simulation/event/telescope/parameters"

# Datalevels
R0_GROUP = "/r0"
R0_TEL_GROUP = "/r0/event/telescope"
R1_GROUP = "/r1"
R1_TEL_GROUP = "/r1/event/telescope"
DL0_GROUP = "/dl0"
DL1_GROUP = "/dl1"
DL1_SUBARRAY_GROUP = "/dl1/event/subarray"
DL1_SUBARRAY_TRIGGER_GROUP = "/dl1/event/subarray/trigger"
DL1_TEL_GROUP = "/dl1/event/telescope"
DL1_TEL_TRIGGER_GROUP = "/dl1/event/telescope/trigger"
DL1_TEL_IMAGES_GROUP = "/dl1/event/telescope/images"
DL1_TEL_PARAMETERS_GROUP = "/dl1/event/telescope/parameters"
DL1_TEL_MUON_GROUP = "/dl1/event/telescope/muon"
DL2_TEL_GROUP = "/dl2/event/telescope"
DL2_GROUP = "/dl2"
DL2_SUBARRAY_GROUP = "/dl2/event/subarray"

# Monitoring group
DL0_TEL_POINTING_GROUP = "/dl0/monitoring/telescope/pointing"
DL1_SUBARRAY_POINTING_GROUP = "/dl1/monitoring/subarray/pointing"
DL1_TEL_POINTING_GROUP = "/dl1/monitoring/telescope/pointing"
DL1_TEL_CALIBRATION_GROUP = "/dl1/monitoring/telescope/calibration"
DL1_CAMERA_MONITORING_GROUP = "/dl1/monitoring/telescope/calibration/camera"
DL1_CAMERA_COEFFICIENTS_GROUP = (
    "/dl1/monitoring/telescope/calibration/camera/coefficients"
)
DL1_PIXEL_STATISTICS_GROUP = (
    "/dl1/monitoring/telescope/calibration/camera/pixel_statistics"
)
DL1_SKY_PEDESTAL_IMAGE_GROUP = (
    "/dl1/monitoring/telescope/calibration/camera/pixel_statistics/sky_pedestal_image"
)
DL1_FLATFIELD_IMAGE_GROUP = (
    "/dl1/monitoring/telescope/calibration/camera/pixel_statistics/flatfield_image"
)
DL1_FLATFIELD_PEAK_TIME_GROUP = (
    "/dl1/monitoring/telescope/calibration/camera/pixel_statistics/flatfield_peak_time"
)

# Column names used for the DL1A data
DL1_COLUMN_NAMES = ["image", "peak_time"]
