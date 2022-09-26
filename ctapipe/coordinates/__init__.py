"""
Coordinates.
"""
import warnings

import astropy.units as u
from astropy.coordinates import (
    CIRS,
    AltAz,
    FunctionTransformWithFiniteDifference,
    frame_transform_graph,
)
from erfa.ufunc import s2p as spherical_to_cartesian

from .camera_frame import CameraFrame, EngineeringCameraFrame
from .ground_frames import (
    EastingNorthingFrame,
    GroundFrame,
    TiltedGroundFrame,
    project_to_ground,
)
from .nominal_frame import NominalFrame
from .telescope_frame import TelescopeFrame

__all__ = [
    "TelescopeFrame",
    "CameraFrame",
    "EngineeringCameraFrame",
    "NominalFrame",
    "GroundFrame",
    "TiltedGroundFrame",
    "EastingNorthingFrame",
    "MissingFrameAttributeWarning",
    "project_to_ground",
    "altaz_to_righthanded_cartesian",
]


class MissingFrameAttributeWarning(Warning):
    pass


# astropy requires all AltAz to have locations
# and obstimes so they can be converted into true skycoords
# Also it is required to transform one AltAz into another one
# This forbids it to use AltAz without setting location and obstime
# here, the astropy behaviour is defined so that it is assumed,
# that if no information about location or obstime is known, both are the same
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, AltAz)
def altaz_to_altaz(from_coo, to_frame):
    # check if coordinates have obstimes defined
    obstime = from_coo.obstime
    if from_coo.obstime is None:
        warnings.warn(
            "AltAz coordinate has no obstime, assuming same frame",
            MissingFrameAttributeWarning,
        )
        obstime = to_frame.obstime

    location = from_coo.location
    if from_coo.obstime is None:
        warnings.warn(
            "Horizontal coordinate has no location, assuming same frame",
            MissingFrameAttributeWarning,
        )
        location = to_frame.location

    if obstime is None or location is None:
        return to_frame.realize_frame(from_coo.data)

    return from_coo.transform_to(CIRS(obstime=obstime)).transform_to(to_frame)


def altaz_to_righthanded_cartesian(alt, az):
    """Turns an alt/az coordinate into a 3d direction in a right-handed coordinate
    system.  This is because azimuths are in a left-handed system.

    See e.g: https://github.com/astropy/astropy/issues/5300

    Parameters
    ----------
    alt: u.Quantity
        altitude
    az: u.Quantity
        azimuth
    """
    if hasattr(az, "unit"):
        az = az.to_value(u.rad)
        alt = alt.to_value(u.rad)

    return spherical_to_cartesian(-az, alt, 1.0)
