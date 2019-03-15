import numpy as np
import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    CoordinateAttribute,
    QuantityAttribute,
    TimeAttribute,
    EarthLocationAttribute,
    FunctionTransform,
    frame_transform_graph,
    CartesianRepresentation,
    UnitSphericalRepresentation,
    AltAz,
)

from .telescope_frame import TelescopeFrame
from .representation import PlanarRepresentation


class CameraFrame(BaseCoordinateFrame):
    '''
    Camera coordinate frame.

    The camera frame is a 2d cartesian frame,
    describing position of objects in the focal plane of the telescope.

    The frame is defined as in H.E.S.S., starting at the horizon,
    the telescope is pointed to magnetic north in azimuth and then up to zenith.

    Now, x points north and y points west, so in this orientation, the
    camera coordinates line up with the CORSIKA ground coordinate system.

    MAGIC and FACT use a different camera coordinate system:
    Standing at the dish, looking at the camera, x points right, y points up.
    To transform MAGIC/FACT to ctapipe, do x' = -y, y' = -x.

    Attributes
    ----------

    focal_length : u.Quantity[length]
        Focal length of the telescope as a unit quantity (usually meters)
    rotation : u.Quantity[angle]
        Rotation angle of the camera (0 deg in most cases)
    telescope_pointing : SkyCoord[AltAz]
        Pointing direction of the telescope as SkyCoord in AltAz
    obstime : Time
        Observation time
    location : EarthLocation
        location of the telescope
    '''
    default_representation = PlanarRepresentation

    focal_length = QuantityAttribute(default=0, unit=u.m)
    rotation = QuantityAttribute(default=0 * u.deg, unit=u.rad)
    telescope_pointing = CoordinateAttribute(frame=AltAz, default=None)

    obstime = TimeAttribute(default=None)
    location = EarthLocationAttribute(default=None)


@frame_transform_graph.transform(FunctionTransform, CameraFrame, TelescopeFrame)
def camera_to_telescope(camera_coord, telescope_frame):
    '''
    Transformation between CameraFrame and TelescopeFrame.
    Is called when a SkyCoord is transformed from CameraFrame into TelescopeFrame
    '''
    x_pos = camera_coord.cartesian.x
    y_pos = camera_coord.cartesian.y

    rot = camera_coord.rotation
    if rot == 0:  # if no rotation applied save a few cycles
        x_rotated = x_pos
        y_rotated = y_pos
    else:
        cosrot = np.cos(rot)
        sinrot = np.sin(rot)
        x_rotated = x_pos * cosrot - y_pos * sinrot
        y_rotated = x_pos * sinrot + y_pos * cosrot

    focal_length = camera_coord.focal_length

    # this assumes an equidistant mapping function of the telescope optics
    # or a small angle approximation of most other mapping functions
    # this could be replaced by actually defining the mapping function
    # as an Attribute of `CameraFrame` that maps f(r, focal_length) -> theta,
    # where theta is the angle to the optical axis and r is the distance
    # to the camera center in the focal plane
    delta_alt = u.Quantity((x_rotated / focal_length).to_value(u.dimensionless_unscaled), u.rad)
    delta_az = u.Quantity((y_rotated / focal_length).to_value(u.dimensionless_unscaled), u.rad)

    representation = UnitSphericalRepresentation(lat=delta_alt, lon=delta_az)

    return telescope_frame.realize_frame(representation)


@frame_transform_graph.transform(FunctionTransform, TelescopeFrame, CameraFrame)
def telescope_to_camera(telescope_coord, camera_frame):
    '''
    Transformation between TelescopeFrame and CameraFrame

    Is called when a SkyCoord is transformed from TelescopeFrame into CameraFrame
    '''
    x_pos = telescope_coord.delta_alt
    y_pos = telescope_coord.delta_az
    # reverse the rotation applied to get to this system
    rot = -camera_frame.rotation

    if rot.value == 0.0:  # if no rotation applied save a few cycles
        x_rotated = x_pos
        y_rotated = y_pos
    else:  # or else rotate all positions around the camera centre
        cosrot = np.cos(rot)
        sinrot = np.sin(rot)
        x_rotated = x_pos * cosrot - y_pos * sinrot
        y_rotated = x_pos * sinrot + y_pos * cosrot

    focal_length = camera_frame.focal_length

    # this assumes an equidistant mapping function of the telescope optics
    # or a small angle approximation of most other mapping functions
    # this could be replaced by actually defining the mapping function
    # as an Attribute of `CameraFrame` that maps f(theta, focal_length) -> r,
    # where theta is the angle to the optical axis and r is the distance
    # to the camera center in the focal plane
    x_rotated = x_rotated.to_value(u.rad) * focal_length
    y_rotated = y_rotated.to_value(u.rad) * focal_length

    representation = CartesianRepresentation(
        x_rotated,
        y_rotated,
        0 * u.m
    )

    return camera_frame.realize_frame(representation)
