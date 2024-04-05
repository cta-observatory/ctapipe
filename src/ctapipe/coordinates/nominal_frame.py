"""
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

__all__ = ["NominalFrame"]


class NominalFrame(BaseCoordinateFrame):
    """
    Nominal coordinate frame.

    A Frame using a UnitSphericalRepresentation.
    This is basically the same as a HorizonCoordinate, but the
    origin is at an arbitrary position in the sky.
    This is what astropy calls a SkyOffsetCoordinate

    If the telescopes are in divergent pointing, this Frame can be
    used to transform to a common system.

    Attributes
    ----------

    origin: astropy.coordinates.SkyCoord[AltAz]
        Origin of this frame as a HorizonCoordinate
    obstime: astropy.time.Time
        Observation time
    location: astropy.coordinates.EarthLocation
        Location of the telescope
    """

    frame_specific_representation_info = {
        UnitSphericalRepresentation: [
            RepresentationMapping("lon", "fov_lon"),
            RepresentationMapping("lat", "fov_lat"),
        ]
    }
    default_representation = UnitSphericalRepresentation

    origin = CoordinateAttribute(default=None, frame=AltAz)

    obstime = TimeAttribute(default=None)
    location = EarthLocationAttribute(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # make sure telescope coordinate is in range [-180°, 180°]
        if isinstance(self._data, UnitSphericalRepresentation):
            self._data.lon.wrap_angle = Angle(180, unit=u.deg)


@frame_transform_graph.transform(FunctionTransform, NominalFrame, NominalFrame)
def nominal_to_nominal(from_nominal_coord, to_nominal_frame):
    """Transform between two skyoffset frames."""

    intermediate_from = from_nominal_coord.transform_to(from_nominal_coord.origin)
    intermediate_to = intermediate_from.transform_to(to_nominal_frame.origin)
    return intermediate_to.transform_to(to_nominal_frame)


@frame_transform_graph.transform(DynamicMatrixTransform, AltAz, NominalFrame)
def altaz_to_nominal(altaz_coord, nominal_frame):
    """Convert a reference coordinate to an sky offset frame."""

    # Define rotation matrices along the position angle vector, and
    # relative to the origin.
    origin = nominal_frame.origin.represent_as(UnitSphericalRepresentation)
    mat1 = rotation_matrix(-origin.lat, "y")
    mat2 = rotation_matrix(origin.lon, "z")
    return mat1 @ mat2


@frame_transform_graph.transform(DynamicMatrixTransform, NominalFrame, AltAz)
def nominal_to_altaz(nominal_coord, altaz_frame):
    """Convert an sky offset frame coordinate to the reference frame"""

    # use the forward transform, but just invert it
    mat = altaz_to_nominal(altaz_frame, nominal_coord)
    # transpose is the inverse because mat is a rotation matrix
    return matrix_transpose(mat)
