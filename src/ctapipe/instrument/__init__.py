"""Describe the telescope subarray, its optics, and its cameras.

Use `SubarrayDescription` to work with telescope positions and instrument
metadata together. Camera geometry, readout settings, and optical models
provide the instrument information needed for event processing.
"""

from .atmosphere import get_atmosphere_profile_functions
from .camera import (
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    PixelGridType,
    PixelShape,
    UnknownPixelShapeWarning,
)
from .guess import guess_telescope
from .optics import (
    ComaPSFModel,
    FocalLengthKind,
    MirrorDescription,
    MirrorFacetShape,
    OpticsDescription,
    PSFModel,
    ReflectorShape,
    SizeType,
    ZernikePSFModel,
)
from .subarray import SubarrayDescription, UnknownSubarray, UnknownTelescopeID
from .telescope import TelescopeDescription
from .trigger import SoftwareTrigger
from .warnings import FromNameWarning

__all__ = [
    "CameraDescription",
    "CameraGeometry",
    "CameraReadout",
    "get_atmosphere_profile_functions",
    "guess_telescope",
    "OpticsDescription",
    "PixelGridType",
    "PixelShape",
    "SubarrayDescription",
    "TelescopeDescription",
    "UnknownTelescopeID",
    "UnknownSubarray",
    "FocalLengthKind",
    "ReflectorShape",
    "SizeType",
    "SoftwareTrigger",
    "FromNameWarning",
    "PSFModel",
    "ComaPSFModel",
    "ZernikePSFModel",
    "UnknownPixelShapeWarning",
    "MirrorDescription",
    "MirrorFacetShape",
]
