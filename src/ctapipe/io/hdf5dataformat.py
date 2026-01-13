"""HDF5 Data Format Constants"""

__all__ = [
    "R0_GROUP",
    "R1_GROUP",
    "DL0_GROUP",
    "DL1_GROUP",
    "DL2_GROUP",
    "SIMULATION_GROUP",
    "SIMULATION_TEL_TABLE",
    "CONFIG_GROUP",
    "CONFIG_INSTRUMENT_SUBARRAY",
    "CONFIG_INSTRUMENT_SUBARRAY_LAYOUT",
    "CONFIG_INSTRUMENT_TEL",
    "CONFIG_INSTRUMENT_TEL_OPTICS",
    "CONFIG_INSTRUMENT_TEL_CAMERA",
    "SCHEDULING_BLOCK_TABLE",
    "OBSERVATION_BLOCK_TABLE",
    "SIMULATION_RUN_TABLE",
    "FIXED_POINTING_GROUP",
    "ATMOSPHERE_DENSITY_PROFILE_TABLE",
    "DL1_IMAGE_STATISTICS_TABLE",
    "DL2_EVENT_STATISTICS_GROUP",
    "SHOWER_DISTRIBUTION_TABLE",
    "SIMULATION_SHOWER_TABLE",
    "SIMULATION_IMPACT_GROUP",
    "SIMULATION_IMAGES_GROUP",
    "SIMULATION_PARAMETERS_GROUP",
    "R0_TEL_GROUP",
    "R1_TEL_GROUP",
    "DL1_SUBARRAY_GROUP",
    "DL1_SUBARRAY_TRIGGER_TABLE",
    "DL1_TEL_GROUP",
    "DL1_TEL_TRIGGER_TABLE",
    "DL1_TEL_IMAGES_GROUP",
    "DL1_TEL_PARAMETERS_GROUP",
    "DL1_TEL_MUON_GROUP",
    "DL2_SUBARRAY_GROUP",
    "DL2_SUBARRAY_GEOMETRY_GROUP",
    "DL2_SUBARRAY_ENERGY_GROUP",
    "DL2_SUBARRAY_PARTICLETYPE_GROUP",
    "DL2_TEL_GROUP",
    "DL2_TEL_GEOMETRY_GROUP",
    "DL2_TEL_ENERGY_GROUP",
    "DL2_TEL_PARTICLETYPE_GROUP",
    "DL0_TEL_POINTING_GROUP",
    "DL1_SUBARRAY_POINTING_GROUP",
    "DL1_TEL_POINTING_GROUP",
    "DL1_TEL_OPTICAL_PSF_GROUP",
    "DL1_TEL_CALIBRATION_GROUP",
    "DL1_TEL_MUON_THROUGHPUT_GROUP",
    "DL1_TEL_ILLUMINATOR_THROUGHPUT_GROUP",
    "DL1_CAMERA_MONITORING_GROUP",
    "DL1_CAMERA_COEFFICIENTS_GROUP",
    "DL1_PIXEL_STATISTICS_GROUP",
    "DL1_PEDESTAL_IMAGE_GROUP",
    "DL1_SKY_PEDESTAL_IMAGE_GROUP",
    "DL1_FLATFIELD_IMAGE_GROUP",
    "DL1_FLATFIELD_PEAK_TIME_GROUP",
    "DL2_SUBARRAY_MONITORING_GROUP",
    "DL2_SUBARRAY_INTER_CALIBRATION_GROUP",
    "DL2_SUBARRAY_CROSS_CALIBRATION_GROUP",
    "DL1_COLUMN_NAMES",
]

# Configuration, service, and simulation group
CONFIG_GROUP = "/configuration"
CONFIG_INSTRUMENT_SUBARRAY = "/configuration/instrument/subarray"
CONFIG_INSTRUMENT_SUBARRAY_LAYOUT = "/configuration/instrument/subarray/layout"
CONFIG_INSTRUMENT_TEL = "/configuration/instrument/telescope"
CONFIG_INSTRUMENT_TEL_OPTICS = "/configuration/instrument/telescope/optics"
CONFIG_INSTRUMENT_TEL_CAMERA = "/configuration/instrument/telescope/camera"
SCHEDULING_BLOCK_TABLE = "/configuration/observation/scheduling_block"
OBSERVATION_BLOCK_TABLE = "/configuration/observation/observation_block"
SIMULATION_RUN_TABLE = "/configuration/simulation/run"
FIXED_POINTING_GROUP = "/configuration/telescope/pointing"

DL1_IMAGE_STATISTICS_TABLE = "/dl1/service/image_statistics"
DL2_EVENT_STATISTICS_GROUP = "/dl2/service/tel_event_statistics"
SIMULATION_GROUP = "/simulation"
SIMULATION_TEL_TABLE = "/simulation/event/telescope"
ATMOSPHERE_DENSITY_PROFILE_TABLE = "/simulation/service/atmosphere_density_profile"
SHOWER_DISTRIBUTION_TABLE = "/simulation/service/shower_distribution"
SIMULATION_SHOWER_TABLE = "/simulation/event/subarray/shower"
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
DL1_SUBARRAY_TRIGGER_TABLE = "/dl1/event/subarray/trigger"
DL1_TEL_GROUP = "/dl1/event/telescope"
DL1_TEL_TRIGGER_TABLE = "/dl1/event/telescope/trigger"
DL1_TEL_IMAGES_GROUP = "/dl1/event/telescope/images"
DL1_TEL_PARAMETERS_GROUP = "/dl1/event/telescope/parameters"
DL1_TEL_MUON_GROUP = "/dl1/event/telescope/muon"
DL2_TEL_GROUP = "/dl2/event/telescope"
DL2_TEL_GEOMETRY_GROUP = "/dl2/event/telescope/geometry"
DL2_TEL_ENERGY_GROUP = "/dl2/event/telescope/energy"
DL2_TEL_PARTICLETYPE_GROUP = "/dl2/event/telescope/particle_type"

DL2_GROUP = "/dl2"
DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
DL2_SUBARRAY_GEOMETRY_GROUP = "/dl2/event/subarray/geometry"
DL2_SUBARRAY_ENERGY_GROUP = "/dl2/event/subarray/energy"
DL2_SUBARRAY_PARTICLETYPE_GROUP = "/dl2/event/subarray/particle_type"

# Monitoring group
DL0_TEL_POINTING_GROUP = "/dl0/monitoring/telescope/pointing"
DL1_SUBARRAY_POINTING_GROUP = "/dl1/monitoring/subarray/pointing"
DL1_TEL_POINTING_GROUP = "/dl1/monitoring/telescope/pointing"
DL1_TEL_OPTICAL_PSF_GROUP = "/dl1/monitoring/telescope/optical_psf"
DL1_TEL_CALIBRATION_GROUP = "/dl1/monitoring/telescope/calibration"
DL1_TEL_MUON_THROUGHPUT_GROUP = (
    "/dl1/monitoring/telescope/calibration/optical_throughput/muon"
)
DL1_TEL_ILLUMINATOR_THROUGHPUT_GROUP = (
    "/dl1/monitoring/telescope/calibration/optical_throughput/illuminator"
)
DL1_CAMERA_MONITORING_GROUP = "/dl1/monitoring/telescope/calibration/camera"
DL1_CAMERA_COEFFICIENTS_GROUP = (
    "/dl1/monitoring/telescope/calibration/camera/coefficients"
)
DL1_PIXEL_STATISTICS_GROUP = (
    "/dl1/monitoring/telescope/calibration/camera/pixel_statistics"
)
DL1_PEDESTAL_IMAGE_GROUP = (
    "/dl1/monitoring/telescope/calibration/camera/pixel_statistics/pedestal_image"
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
DL2_SUBARRAY_MONITORING_GROUP = "/dl2/monitoring/subarray"
DL2_SUBARRAY_INTER_CALIBRATION_GROUP = "/dl2/monitoring/subarray/inter_calibration"
DL2_SUBARRAY_CROSS_CALIBRATION_GROUP = "/dl2/monitoring/subarray/cross_calibration"

# Column names used for the DL1A data
DL1_COLUMN_NAMES = ["image", "peak_time"]
