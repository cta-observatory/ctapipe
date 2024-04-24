"""
Coordinates.
"""
import warnings

from astropy.coordinates import (
    CIRS,
    AltAz,
    FunctionTransformWithFiniteDifference,
    frame_transform_graph,
)

from .camera_frame import CameraFrame, EngineeringCameraFrame
from .ground_frames import (
    EastingNorthingFrame,
    GroundFrame,
    TiltedGroundFrame,
    project_to_ground,
)
from .impact_distance import impact_distance, shower_impact_distance
from .nominal_frame import NominalFrame
from .telescope_frame import TelescopeFrame
from .utils import altaz_to_righthanded_cartesian, get_point_on_shower_axis

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
    "impact_distance",
    "shower_impact_distance",
    "get_point_on_shower_axis",
]


class MissingFrameAttributeWarning(Warning):
    pass


def get_representation_component_names(frame):
    """Return the component names of a Frame (or SkyCoord)"""
    return tuple(frame.get_representation_component_names().keys())


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
            "AltAz coordinate has no location, assuming same frame",
            MissingFrameAttributeWarning,
        )
        location = to_frame.location

    if obstime is None or location is None:
        return to_frame.realize_frame(from_coo.data)

    return from_coo.transform_to(CIRS(obstime=obstime)).transform_to(to_frame)
