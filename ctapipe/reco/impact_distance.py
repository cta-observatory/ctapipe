#!/usr/bin/env python3

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, spherical_to_cartesian

from ..containers import ReconstructedGeometryContainer
from ..coordinates import GroundFrame, TiltedGroundFrame
from ..instrument.subarray import SubarrayDescription


__all__ = ["shower_impact_distance"]


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
    shower_geom: ReconstructedGeometryContainer, subarray: SubarrayDescription
):
    """computes the impact distance of the shower axis to the telescope positions

    Parameters
    ----------
    shower_geom: ReconstructedGeometryContainer
        reconstructed shower geometry, must contain a core and alt/az position
    subarray: SubarrayDescription
        SubarrayDescription from which to extract the telescope positions

    Returns
    -------
    np.ndarray:
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
    # NOTE: remember that in sky direction the azimuth needs to be negative
    sky_direction = spherical_to_cartesian(
        r=1, lat=shower_geom.alt, lon=-shower_geom.az
    )

    # telescope positions in the ground frame
    telescope_positions = subarray.tel_coords.cartesian.xyz.to_value("m").T

    return (
        impact_distance(
            point=core_position,
            direction=sky_direction,
            test_points=telescope_positions,
        )
        * u.m
    )


def shower_impact_distance_with_frames(
    shower_geom: ReconstructedGeometryContainer, subarray: SubarrayDescription
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
