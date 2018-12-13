"""
Coordinates.
"""
from .horizon_frame import HorizonFrame
from .telescope_frame import TelescopeFrame
from .nominal_frame import NominalFrame
from .ground_frames import GroundFrame, TiltedGroundFrame, project_to_ground
from .camera_frame import CameraFrame

from astropy.utils import iers
iers.conf.auto_download = True  # auto-fetch updates to IERS_A table


__all__ = [
    'CameraFrame',
    'TelescopeFrame',
    'NominalFrame',
    'HorizonFrame',
    'GroundFrame',
    'TiltedGroundFrame',
    'project_to_ground',
    'transformations'
]
