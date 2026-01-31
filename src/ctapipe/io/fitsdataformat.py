"""FITS Data Format Constants for CTAO DL0 Monitoring Data

This module defines constants for FITS file structure including header keywords,
column names, and other format-specific values used in CTAO DL0 monitoring data.

Note: These constants are based on the CTAO DL0 Data Model specification.
As the detailed FITS format specification is finalized, these values should
be updated accordingly.
"""

from typing import NamedTuple

__all__ = [
    "SOURCE",
    "NAME",
    "DESCRIPTION",
    "LST_DRIVE_DATA_STRUCT",
    "FRAM_DATA_STRUCT",
    "MonitoringSource",
    "LST1_DRIVE",
    "STPN_01",
    "MONITORING_SOURCES",
]

# FITS Header Keywords
# ---------------------
SOURCE = "MSOURCE"
NAME = "MNAME"
DESCRIPTION = "MDESC"


# FITS Table Data Structures
# ---------------------------
# Dictionaries mapping column names to their descriptions

# Telescope Pointing columns
LST_DRIVE_DATA_STRUCT = {
    "time": "LowPrecisionTime timestamp of the pointing measurement",
    "altitude": "Altitude angle",
    "azimuth": "Azimuth angle",
}

# FRAM monitoring data columns
FRAM_DATA_STRUCT = {
    "timestamp": "Measurement timestamp",
    "ra": "Array of Voronoi cell centers RA coordinates",
    "dec": "Array of Voronoi points centers DEC coordinates",
    "aod": "Array of aerosol optical depth values for each Voronoi cell",
    "aod_err": "Array of errors associated with each AOD measurement",
}


class MonitoringSource(NamedTuple):
    """Monitoring source definition."""

    name: str
    data_struct: dict[str, str]
    description: str = ""


# Define monitoring sources
LST1_DRIVE = MonitoringSource(
    name="LST1Drive",
    data_struct=LST_DRIVE_DATA_STRUCT,
    description="LST-1 telescope drive pointing",
)

STPN_01 = MonitoringSource(
    name="STPN-01",
    data_struct=FRAM_DATA_STRUCT,
    description="FRAM atmospheric monitoring",
)

# Registry mapping source names to definitions
MONITORING_SOURCES = {
    "LST1Drive": LST1_DRIVE,
    "STPN-01": STPN_01,
}
