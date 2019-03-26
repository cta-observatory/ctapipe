'''
Coordinates.
'''
from astropy.coordinates import (
    AltAz,
    FunctionTransformWithFiniteDifference,
    CIRS,
    frame_transform_graph,
)
import warnings
from .telescope_frame import TelescopeFrame
from .nominal_frame import NominalFrame
from .ground_frames import GroundFrame, TiltedGroundFrame, project_to_ground
from .camera_frame import CameraFrame


__all__ = [
    'TelescopeFrame',
    'CameraFrame',
    'NominalFrame',
    'GroundFrame',
    'TiltedGroundFrame',
    'project_to_ground',
    'MissingFrameAttributeWarning',
]


class MissingFrameAttributeWarning(Warning):
    pass


# astropy requires all AltAz to have locations
# and obstimes so they can be converted into true skycoords
# Also it is required to transform one AltAz into another one
# This forbids it to use AltAz without setting location and obstime
# here, the astropy behaviour is defined so that it is assumed,
# that if no information about location or obstime is known, both are the same
@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    AltAz,
    AltAz
)
def altaz_to_altaz(from_coo, to_frame):
    # check if coordinates have obstimes defined
    obstime = from_coo.obstime
    if from_coo.obstime is None:
        warnings.warn(
            'AltAz coordinate has no obstime, assuming same frame',
            MissingFrameAttributeWarning,
        )
        obstime = to_frame.obstime

    location = from_coo.location
    if from_coo.obstime is None:
        warnings.warn(
            'Horizontal coordinate has no location, assuming same frame',
            MissingFrameAttributeWarning,
        )
        location = to_frame.location

    if obstime is None or location is None:
        return to_frame.realize_frame(from_coo.spherical)

    return from_coo.transform_to(CIRS(obstime=obstime)).transform_to(to_frame)
