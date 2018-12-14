'''
Coordinates.
'''
from .horizon_frame import HorizonFrame
from .telescope_frame import TelescopeFrame
from .nominal_frame import NominalFrame
from .ground_frames import GroundFrame, TiltedGroundFrame, project_to_ground
from .camera_frame import CameraFrame

from astropy.utils import iers
iers.conf.auto_download = True  # auto-fetch updates to IERS_A table


# HorizonFrame is not in __all__ because the docs complain otherwise
__all__ = [
    'TelescopeFrame',
    'CameraFrame',
    'NominalFrame',
    'GroundFrame',
    'TiltedGroundFrame',
    'project_to_ground',
]
