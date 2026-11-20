"""
Define the telescope coordinate frame.

The code in this module is basically a copy of
https://docs.astropy.org/en/stable/_modules/astropy/coordinates/builtin_frames/skyoffset.html

We are just not creating a metaclass and a factory but directly building the
corresponding class.
"""

import astropy.units as u
from astropy.coordinates import (
    AltAz,
    Angle,
    BaseCoordinateFrame,
    CoordinateAttribute,
    DynamicMatrixTransform,
    EarthLocationAttribute,
    FunctionTransform,
    RepresentationMapping,
    TimeAttribute,
    UnitSphericalRepresentation,
    frame_transform_graph,
)
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix

__all__ = ["TelescopeFrame"]


_wrap_angle = Angle(180, unit=u.deg)


class TelescopeFrame(BaseCoordinateFrame):
    """
    Telescope coordinate frame.

    A Frame using a UnitSphericalRepresentation.
    This is basically the same as a HorizonCoordinate, but the
    origin is at the telescope's pointing direction.

    This is used to specify coordinates in the field of view of a
    telescope that is independent of the optical properties of the telescope.

    ``fov_lon`` is aligned with azimuth and ``fov_lat`` is aligned with altitude
    of the horizontal coordinate frame as implemented in ``astropy.coordinates.AltAz``.

    This is what astropy calls a SkyOffsetCoordinate.

    Parameters
    ----------
    *args
        Coordinate components, or a coordinate representation, passed to Astropy.
    **kwargs
        Frame options passed to Astropy, including ``telescope_pointing`` in AltAz,
        ``obstime``, and ``location``.

    Attributes
    ----------
    telescope_pointing: SkyCoord[AltAz]
        Coordinate of the telescope pointing in AltAz
    obstime: Time
        Observation time
    location: EarthLocation
        Location of the telescope
    """

    frame_specific_representation_info = {
        UnitSphericalRepresentation: [
            RepresentationMapping("lon", "fov_lon"),
            RepresentationMapping("lat", "fov_lat"),
        ]
    }
    default_representation = UnitSphericalRepresentation

    telescope_pointing = CoordinateAttribute(default=None, frame=AltAz)

    obstime = TimeAttribute(default=None)
    location = EarthLocationAttribute(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # make sure telescope coordinate is in range [-180°, 180°]
        if isinstance(self._data, UnitSphericalRepresentation):
            self._data.lon.wrap_angle = _wrap_angle


@frame_transform_graph.transform(FunctionTransform, TelescopeFrame, TelescopeFrame)
def telescope_to_telescope(from_telescope_coord, to_telescope_frame):
    """Transform between two skyoffset frames.

    Parameters
    ----------
    from_telescope_coord : TelescopeFrame
        Input field-of-view coordinates relative to the original telescope pointing.
    to_telescope_frame : TelescopeFrame
        Target frame supplying the new telescope pointing.
    """
    intermediate_from = from_telescope_coord.transform_to(
        from_telescope_coord.telescope_pointing
    )
    intermediate_to = intermediate_from.transform_to(
        to_telescope_frame.telescope_pointing
    )
    return intermediate_to.transform_to(to_telescope_frame)


@frame_transform_graph.transform(DynamicMatrixTransform, AltAz, TelescopeFrame)
def altaz_to_telescope(altaz_coord, telescope_frame):
    """Convert a reference coordinate to an sky offset frame.

    Parameters
    ----------
    altaz_coord : astropy.coordinates.AltAz
        Source frame required by the rotation interface. The rotation is determined by
        the telescope pointing.
    telescope_frame : TelescopeFrame
        Target frame supplying the pointing direction that defines the rotation.
    """
    # Define rotation matrices along the position angle vector, and
    # relative to the telescope_pointing.
    telescope_pointing = telescope_frame.telescope_pointing.represent_as(
        UnitSphericalRepresentation
    )
    mat1 = rotation_matrix(-telescope_pointing.lat, "y")
    mat2 = rotation_matrix(telescope_pointing.lon, "z")
    return mat1 @ mat2


@frame_transform_graph.transform(DynamicMatrixTransform, TelescopeFrame, AltAz)
def telescope_to_altaz(telescope_coord, altaz_frame):
    """Convert an sky offset frame coordinate to the reference frame

    Parameters
    ----------
    telescope_coord : TelescopeFrame
        Source frame whose pointing direction defines the inverse rotation.
    altaz_frame : astropy.coordinates.AltAz
        Target horizontal coordinate frame.
    """
    # use the forward transform, but just invert it
    mat = altaz_to_telescope(altaz_frame, telescope_coord)
    # transpose is the inverse because mat is a rotation matrix
    return matrix_transpose(mat)
