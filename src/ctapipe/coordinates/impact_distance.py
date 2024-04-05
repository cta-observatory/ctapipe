"""
Functions to compute the impact distance from a simulated or reconstructed
shower axis (Defined by the line from the impact point on the ground in the
reconstructed sky direction) to each telescope's ground position.
"""
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from ..containers import ReconstructedGeometryContainer, SimulatedShowerContainer
from .ground_frames import GroundFrame, TiltedGroundFrame
from .utils import altaz_to_righthanded_cartesian

__all__ = ["shower_impact_distance", "impact_distance"]


def impact_distance(point: np.ndarray, direction: np.ndarray, test_points: np.ndarray):
    """Compute impact distance from a line defined by a point and a direction vector
    with an array of points

    Parameters
    ----------
    point: np.ndarray
       point defining the line
    direction: np.ndarray
       direction vector
    test_points: np.ndarray
       array of test points
    """
    return np.linalg.norm(
        np.cross(test_points - point, direction), axis=1
    ) / np.linalg.norm(direction)


def shower_impact_distance(
    shower_geom: ReconstructedGeometryContainer | SimulatedShowerContainer,
    subarray,
):
    """computes the impact distance of the shower axis to the telescope positions

    Parameters
    ----------
    shower_geom : ReconstructedGeometryContainer
        reconstructed shower geometry, must contain a core and alt/az position
    subarray : SubarrayDescription
        SubarrayDescription from which to extract the telescope positions

    Returns
    -------
    Quantity[m] :
       array of impact distances to each telescope (packed by tel index)
    """

    # core position in the ground frame
    core_position = np.array(
        [
            shower_geom.core_x.to_value(u.m),
            shower_geom.core_y.to_value(u.m),
            0,
        ]
    )

    # sky direction of the shower (defines the shower axis)
    sky_direction = altaz_to_righthanded_cartesian(
        alt=shower_geom.alt, az=shower_geom.az
    )

    # telescope positions in the ground frame
    telescope_positions = subarray.tel_coords.cartesian.xyz.to_value(u.m).T

    return u.Quantity(
        impact_distance(
            point=core_position,
            direction=sky_direction,
            test_points=telescope_positions,
        ),
        u.m,
        copy=False,
    )


def shower_impact_distance_with_frames(
    shower_geom: ReconstructedGeometryContainer, subarray
):
    """an alternate and slower implementation for cross-check testing purposes.

    Here we make a TiltedFrame that is aligned with the shower axis and
    transform both the telescopes and the impact point into that frame. Then the
    cartesian distance in the X-Y plane should be the impact distance.
    """

    tilted = TiltedGroundFrame(
        pointing_direction=SkyCoord(
            alt=shower_geom.alt, az=shower_geom.az, frame="altaz"
        )
    )

    core_pos_shower_tilted = (
        SkyCoord(
            x=shower_geom.core_x,
            y=shower_geom.core_y,
            z=0 * u.m,
            frame=GroundFrame(),
        )
        .transform_to(tilted)
        .cartesian.xyz
    )

    tel_pos_shower_tilted = subarray.tel_coords.transform_to(tilted).cartesian.xyz.T

    # compute cartesian distance in the tilted frame (only x,y coordinates needed):
    delta = tel_pos_shower_tilted - core_pos_shower_tilted
    return np.hypot(delta[:, 0], delta[:, 1])
