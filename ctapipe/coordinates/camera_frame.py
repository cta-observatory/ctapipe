import numpy as np
import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    CoordinateAttribute,
    QuantityAttribute,
    Attribute,
    TimeAttribute,
    EarthLocationAttribute,
    FunctionTransform,
    frame_transform_graph,
    CartesianRepresentation,
    UnitSphericalRepresentation,
    AltAz,
    AffineTransform,
)

from .telescope_frame import TelescopeFrame
from .representation import PlanarRepresentation


def barrel_distortion(x, y, alpha=0):
    """Apply a barrel distortion to a set of coordinates.

    Parameters
    ----------
    x : array_like
        Cartesian coordinates along the 'x' axis.
        Shape: (n_pixels)
    y : array_like
        Cartesian coordinates along the 'y' axis.
        Shape: (n_pixels)
    alpha : float
        Correction factor for which negative values mean shrinking.


    Returns
    -------
    xp : array_like
        Cartesian coordinates along the 'x' axis corrected for barrel distortion.
        Shape: (n_pixels)
    yp : array_like
        Cartesian coordinates along the 'y' axis corrected for barrel distortion.
        Shape: (n_pixels)

    M.Peresano, K.Kosack, 2020
    """
    r = np.hypot(x, y).value  # pixel-wise radii
    xp = x / (1.0 - alpha * r)
    yp = y / (1.0 - alpha * r)
    return xp, yp


def apply_coma_correction(pix_x, pix_y, alpha=0):
    """Apply correction for Coma aberration to a set of pixel coordinates.

    This aberration shifts the focused light further away from the center of
    the camera. The pixels must be moved back before parametrizing to image.
    The radial shift is proportional to the distance from the center.

    Parameters
    ----------
    x : array_like
        Cartesian coordinates of the pixels along the 'x' axis.
        Shape: (n_pixels)
    y : array_like
        Cartesian coordinates of the pixels along the 'y' axis.
        Shape: (n_pixels)
    alpha : float
        Correction factor equivalent to (1-f) where f is the ratio between
        effective and nominal focal length of the telescope
        (https://www.mpi-hd.mpg.de/hfm/CTA/MC/Prod3/Config/PSF/flen.pdf)
        In this way alpha > 0 means positive distortion.

    Returns
    -------
    x_coma : array_like
        Cartesian coordinates of the pixels along the 'x' axis corrected for
        Coma aberration.
        Shape: (n_pixels)
    y_coma : array_like
        Cartesian coordinates of the pixels along the 'y' axis corrected for
        Coma aberration.
        Shape: (n_pixels)

    M.Peresano, K.Kosack, 2020
    """
    # Take the original pixel-coordinates and correct them
    x_coma, y_coma = barrel_distortion(pix_x, pix_y, alpha=alpha)
    return x_coma, y_coma


class MirrorAttribute(Attribute):
    """A frame Attribute that can only store the integers 1 and 2"""

    def convert_input(self, value):
        """make sure input is 1 or 2"""
        if value in (1, 2):
            return value, False

        raise ValueError("Only 1 or 2 mirrors supported")


# Go from SimTel / HESS to MAGIC/FACT/Engineering frame and back
CAMERA_TO_ENGINEERING_1M_MATRIX = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
ENGINEERING_1M_TO_CAMERA_MATRIX = CAMERA_TO_ENGINEERING_1M_MATRIX
CAMERA_TO_ENGINEERING_2M_MATRIX = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
ENGINEERING_2M_TO_CAMERA_MATRIX = CAMERA_TO_ENGINEERING_2M_MATRIX.T
ZERO_OFFSET = CartesianRepresentation(0, 0, 0, unit=u.m)


class CameraFrame(BaseCoordinateFrame):
    """
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
    coma_correction: float
        Scale factor to correct for Coma aberrations equivalent to (1-f) with f
        the ratio between effective and nominal focal length of the telescope
        (https://www.mpi-hd.mpg.de/hfm/CTA/MC/Prod3/Config/PSF/flen.pdf).
        In this way alpha < 0 means shrinking.
    obstime : Time
        Observation time
    location : EarthLocation
        location of the telescope
    """

    default_representation = PlanarRepresentation

    focal_length = QuantityAttribute(default=0, unit=u.m)
    rotation = QuantityAttribute(default=0 * u.deg, unit=u.rad)
    coma_correction = 0  # no pixel moves towards the camera center
    telescope_pointing = CoordinateAttribute(frame=AltAz, default=None)

    obstime = TimeAttribute(default=None)
    location = EarthLocationAttribute(default=None)


class EngineeringCameraFrame(CameraFrame):
    """
    Engineering camera coordinate frame.

    The camera frame is a 2d cartesian frame,
    describing position of objects in the focal plane of the telescope.

    The frame is defined as in MAGIC and FACT.
    Standing in the dish, looking onto the camera, x points right and y points up.

    HESS, ctapipe and sim_telarray use a different camera coordinate system:
    To transform H.E.S.S./ctapipe -> FACT/MAGIC, do x' = -y, y' = -x.

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
    """

    n_mirrors = MirrorAttribute(default=1)


@frame_transform_graph.transform(FunctionTransform, CameraFrame, TelescopeFrame)
def camera_to_telescope(camera_coord, telescope_frame):
    """
    Transformation between CameraFrame and TelescopeFrame.
    Is called when a SkyCoord is transformed from CameraFrame into TelescopeFrame
    """
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

    coma = camera_coord.coma_correction
    if coma == 0:  # if no Coma correction applied save a few cycles
        x_coma = x_rotated
        y_coma = y_rotated
    else:
        x_coma, y_coma = apply_coma_correction(x_rotated, y_rotated, coma)

    focal_length = camera_coord.focal_length

    # this assumes an equidistant mapping function of the telescope optics
    # or a small angle approximation of most other mapping functions
    # this could be replaced by actually defining the mapping function
    # as an Attribute of `CameraFrame` that maps f(r, focal_length) -> theta,
    # where theta is the angle to the optical axis and r is the distance
    # to the camera center in the focal plane
    delta_alt = u.Quantity(
        (x_coma / focal_length).to_value(u.dimensionless_unscaled), u.rad
    )
    delta_az = u.Quantity(
        (y_coma / focal_length).to_value(u.dimensionless_unscaled), u.rad
    )

    representation = UnitSphericalRepresentation(lat=delta_alt, lon=delta_az)

    return telescope_frame.realize_frame(representation)


@frame_transform_graph.transform(FunctionTransform, TelescopeFrame, CameraFrame)
def telescope_to_camera(telescope_coord, camera_frame):
    """
    Transformation between TelescopeFrame and CameraFrame

    Is called when a SkyCoord is transformed from TelescopeFrame into CameraFrame
    """
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

    # apply backwards the Coma correction used to get to this system
    coma = -camera_frame.coma_correction  # this should now be positive
    if coma == 0:  # if no Coma correction applied save a few cycles
        x_coma = x_rotated
        y_coma = y_rotated
    else:  # or else move all pixels outwards from the camera centre
        x_coma, y_coma = apply_coma_correction(x_rotated, y_rotated, coma)

    focal_length = camera_frame.focal_length

    # this assumes an equidistant mapping function of the telescope optics
    # or a small angle approximation of most other mapping functions
    # this could be replaced by actually defining the mapping function
    # as an Attribute of `CameraFrame` that maps f(theta, focal_length) -> r,
    # where theta is the angle to the optical axis and r is the distance
    # to the camera center in the focal plane
    x_coma = x_coma.to_value(u.rad) * focal_length
    y_coma = y_coma.to_value(u.rad) * focal_length

    representation = CartesianRepresentation(x_coma, y_coma, 0 * u.m)

    return camera_frame.realize_frame(representation)


@frame_transform_graph.transform(AffineTransform, CameraFrame, EngineeringCameraFrame)
def camera_to_engineering(from_coord, to_frame):
    if to_frame.n_mirrors == 1:
        return CAMERA_TO_ENGINEERING_1M_MATRIX, ZERO_OFFSET

    return CAMERA_TO_ENGINEERING_2M_MATRIX, ZERO_OFFSET


@frame_transform_graph.transform(AffineTransform, EngineeringCameraFrame, CameraFrame)
def engineering_to_camera(from_coord, to_frame):
    if from_coord.n_mirrors == 1:
        return ENGINEERING_1M_TO_CAMERA_MATRIX, ZERO_OFFSET
    return ENGINEERING_2M_TO_CAMERA_MATRIX, ZERO_OFFSET
