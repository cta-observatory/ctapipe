"""Describe camera pixel geometry and waveform readout settings.

Geometry describes the layout, shapes, and adjacency of pixels. Readout
describes waveform sampling and the reference pulse shapes.
"""

from .description import CameraDescription  # noqa: F401
from .geometry import (
    CameraGeometry,  # noqa: F401
    PixelGridType,  # noqa: F401
    PixelShape,  # noqa: F401
    UnknownPixelShapeWarning,  # noqa: F401
)
from .readout import CameraReadout  # noqa: F401

# commented out due to sphinx issue with classes being defined in 3 places
__all__ = [
    # "CameraDescription",
    # "CameraGeometry",
    # "PixelShape",
    # "UnknownPixelShapeWarning",
    # "CameraReadout",
]
