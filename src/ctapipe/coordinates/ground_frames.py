"""This module defines the important coordinate systems to be used in
reconstruction with the CTA pipeline and the transformations between
this different systems. Frames and transformations are defined using
the astropy.coordinates framework. This module defines transformations
for ground based cartesian and planar systems.
"""
import astropy.units as u
import numpy as np
from astropy.coordinates import (
    ITRS,
    AffineTransform,
    AltAz,
    BaseCoordinateFrame,
    CartesianRepresentation,
    CoordinateAttribute,
    EarthLocationAttribute,
    FunctionTransform,
    RepresentationMapping,
    frame_transform_graph,
)
from astropy.units.quantity import Quantity

__all__ = [
    "GroundFrame",
    "TiltedGroundFrame",
    "project_to_ground",
    "EastingNorthingFrame",
]


def _altaz_to_earthlocation(altaz):
    local_itrs = altaz.transform_to(ITRS(location=altaz.location))
    itrs = ITRS(local_itrs.cartesian + altaz.location.get_itrs().cartesian)
    return itrs.earth_location


def _earthlocation_to_altaz(location, reference_location):
    # See
    # https://docs.astropy.org/en/stable/coordinates/common_errors.html#altaz-calculations-for-earth-based-objects
    # for why this is necessary and we cannot just do
    # `get_itrs().transform_to(AltAz())`
    itrs_cart = location.get_itrs().cartesian
    itrs_ref_cart = reference_location.get_itrs().cartesian
    local_itrs = ITRS(itrs_cart - itrs_ref_cart, location=reference_location)
    return local_itrs.transform_to(AltAz(location=reference_location))


class GroundFrame(BaseCoordinateFrame):
    """Ground coordinate frame.

    The ground coordinate frame is a simple cartesian frame describing the
    3 dimensional position of objects compared to the array ground level
    in relation to the nominal center of the array.

    Typically this frame will be used for describing the position of telescopes,
    equipment and shower impact coordinates.

    In this frame x points north, y points west and z is meters above array center.

    Frame attributes: None

    """

    default_representation = CartesianRepresentation
    reference_location = EarthLocationAttribute()

    def to_earth_location(self):
        """Convert this GroundFrame coordinate into an `astropy.coordinates.EarthLocation`

        This requires that the ``reference_location`` is set.
        """
        # in astropy, x points north, y points east, so we need a minus for y.
        cart = CartesianRepresentation(self.x, -self.y, self.z)
        altaz = AltAz(cart, location=self.reference_location)
        return _altaz_to_earthlocation(altaz)

    @classmethod
    def from_earth_location(cls, location, reference_location):
        """
        Convert `~astropy.coordinates.EarthLocation` into Groundframe.

        Parameters
        ----------
        location : astropy.coordinates.EarthLocation
            The location to convert
        reference_location : astropy.coordinates.EarthLocation
            Reference location for the GroundFrame

        Returns
        -------
        ground_frame : GroundFrame
            EarthLocation converted to GroundFrame
        """
        altaz = _earthlocation_to_altaz(location, reference_location)
        x, y, z = altaz.cartesian.xyz
        # in astropy, x points north, y points east, so we need a minus for y.
        return GroundFrame(x=x, y=-y, z=z, reference_location=reference_location)

    @property
    def observation_level(self):
        """Height of the reference location"""
        return self.reference_location.height


class EastingNorthingFrame(BaseCoordinateFrame):
    """GroundFrame but in standard Easting/Northing coordinates instead of
    SimTel/Corsika conventions

    Frame attributes: None

    """

    default_representation = CartesianRepresentation
    reference_location = EarthLocationAttribute()

    frame_specific_representation_info = {
        CartesianRepresentation: [
            RepresentationMapping("x", "easting"),
            RepresentationMapping("y", "northing"),
            RepresentationMapping("z", "height"),
        ]
    }

    def to_earth_location(self):
        # in astropy, x points north, y points east
        cart = CartesianRepresentation(self.northing, self.easting, self.height)
        altaz = AltAz(cart, location=self.reference_location)
        return _altaz_to_earthlocation(altaz)

    @classmethod
    def from_earth_location(cls, location, reference_location):
        altaz = _earthlocation_to_altaz(location, reference_location)
        x, y, z = altaz.cartesian.xyz
        # in astropy, x points north, y points east
        return GroundFrame(
            northing=x,
            easting=y,
            height=z,
            reference_location=reference_location,
        )

    @property
    def observation_level(self):
        return self.reference_location.height


class TiltedGroundFrame(BaseCoordinateFrame):
    """Tilted ground coordinate frame.  The tilted ground coordinate frame
    is a cartesian system describing the 2 dimensional projected
    positions of objects in a tilted plane described by
    pointing_direction Typically this frame will be used for the
    reconstruction of the shower core position

    Frame attributes:

    * ``pointing_direction``
        Alt,Az direction of the tilted reference plane

    """

    default_representation = CartesianRepresentation
    # Pointing direction of the tilted system (alt,az),
    # could be the telescope pointing direction or the reconstructed shower
    # direction
    pointing_direction = CoordinateAttribute(default=None, frame=AltAz)


def _get_shower_trans_matrix(azimuth, altitude, inverse=False):
    """Get Transformation matrix for conversion from the ground system to
    the Tilted system and back again (This function is directly lifted
    from read_hess, probably could be streamlined using python
    functionality)

    Parameters
    ----------
    azimuth: float or ndarray
        Azimuth angle in radians of the tilted system used
    altitude: float or ndarray
        Altitude angle in radiuan of the tilted system used

    Returns
    -------
    trans: 3x3 ndarray transformation matrix
    """
    cos_z = np.sin(altitude)  # this is the same as np.cos(zenith) but faster
    sin_z = np.cos(altitude)
    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)

    trans = np.array(
        [
            [cos_z * cos_az, -cos_z * sin_az, -sin_z],
            [sin_az, cos_az, np.zeros_like(sin_z)],
            [sin_z * cos_az, -sin_z * sin_az, cos_z],
        ],
        dtype=np.float64,
    )

    if inverse:
        return np.swapaxes(trans, 0, 1)

    return trans


def _get_xyz(coord):
    """
    Essentially the same as coord.cartesian.xyz, but much faster by
    avoiding some astropy bottlenecks.
    """
    # this is a speed optimization. Much faster to use data if already a
    # Cartesian object
    if isinstance(coord.data, CartesianRepresentation):
        cart = coord.data
    else:
        cart = coord.cartesian

    # this is ~5x faster then cart.xyz
    return u.Quantity([cart.x, cart.y, cart.z])


@frame_transform_graph.transform(FunctionTransform, GroundFrame, TiltedGroundFrame)
def ground_to_tilted(ground_coord, tilted_frame):
    """
    Transformation from ground system to tilted ground system

    Parameters
    ----------
    ground_coord: `astropy.coordinates.SkyCoord`
        Coordinate in GroundFrame
    tilted_frame: `ctapipe.coordinates.TiltedFrame`
        Frame to transform to

    Returns
    -------
    SkyCoordinate transformed to `tilted_frame` coordinates
    """
    xyz_grd = _get_xyz(ground_coord)

    # convert to rad first and subtract. Faster than .zen
    altitude = tilted_frame.pointing_direction.alt.to_value(u.rad)
    azimuth = tilted_frame.pointing_direction.az.to_value(u.rad)

    rotation_matrix = _get_shower_trans_matrix(azimuth, altitude)

    vec = np.einsum("ij...,j...->i...", rotation_matrix, xyz_grd)

    representation = CartesianRepresentation(*vec)

    return tilted_frame.realize_frame(representation)


@frame_transform_graph.transform(FunctionTransform, TiltedGroundFrame, GroundFrame)
def tilted_to_ground(tilted_coord, ground_frame):
    """
    Transformation from tilted ground system to  ground system

    Parameters
    ----------
    tilted_coord: `astropy.coordinates.SkyCoord`
        TiltedGroundFrame system
    ground_frame: `astropy.coordinates.SkyCoord`
        GroundFrame system

    Returns
    -------
    GroundFrame coordinates
    """
    xyz_tilt = _get_xyz(tilted_coord)

    altitude = tilted_coord.pointing_direction.alt.to_value(u.rad)
    azimuth = tilted_coord.pointing_direction.az.to_value(u.rad)

    rotation_matrix = _get_shower_trans_matrix(azimuth, altitude, inverse=True)

    vec = np.einsum("ij...,j...->i...", rotation_matrix, xyz_tilt)

    representation = CartesianRepresentation(*vec)

    grd = ground_frame.realize_frame(representation)

    return grd


def project_to_ground(tilt_system):
    """Project position in the tilted system onto the ground. This is
    needed as the standard transformation will return the 3d position
    of the tilted frame. This projection may ultimately be the
    standard use case so may be implemented in the tilted to ground
    transformation

    Parameters
    ----------
    tilt_system: `astropy.coordinates.SkyCoord`
        coordinate in the the tilted ground system

    Returns
    -------
    Projection of tilted system onto the ground (GroundSystem)

    """
    ground_system = tilt_system.transform_to(GroundFrame)

    unit = ground_system.x.unit
    x_initial = ground_system.x.value
    y_initial = ground_system.y.value
    z_initial = ground_system.z.value

    trans = _get_shower_trans_matrix(
        tilt_system.pointing_direction.az.to_value(u.rad),
        tilt_system.pointing_direction.alt.to_value(u.rad),
    )

    x_projected = x_initial - trans[2][0] * z_initial / trans[2][2]
    y_projected = y_initial - trans[2][1] * z_initial / trans[2][2]

    return GroundFrame(
        x=u.Quantity(x_projected, unit),
        y=u.Quantity(y_projected, unit),
        z=u.Quantity(0, unit),
    )


@frame_transform_graph.transform(FunctionTransform, GroundFrame, GroundFrame)
def ground_to_ground(ground_coords, ground_frame):
    """Null transform for going from ground to ground, since there are no
    attributes of the GroundSystem"""
    return ground_coords


# Matrices for transforming between GroundFrame and EastingNorthingFrame
NO_OFFSET = CartesianRepresentation(Quantity([0, 0, 0], u.m))
GROUND_TO_EASTNORTH = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])


@frame_transform_graph.transform(AffineTransform, GroundFrame, EastingNorthingFrame)
def ground_to_easting_northing(ground_coords, eastnorth_frame):
    """
    convert GroundFrame points into eastings/northings for plotting purposes

    """

    return GROUND_TO_EASTNORTH, NO_OFFSET


@frame_transform_graph.transform(AffineTransform, EastingNorthingFrame, GroundFrame)
def easting_northing_to_ground(eastnorth_coords, ground_frame):
    """
    convert  eastings/northings back to GroundFrame

    """
    return GROUND_TO_EASTNORTH.T, NO_OFFSET
