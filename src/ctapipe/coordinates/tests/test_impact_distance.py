#!/usr/bin/env python3

"""
tests for impact distance computations
"""

import numpy as np
from astropy import units as u

from ctapipe.containers import ReconstructedGeometryContainer
from ctapipe.coordinates.impact_distance import (
    impact_distance,
    shower_impact_distance,
    shower_impact_distance_with_frames,
)
from ctapipe.instrument.subarray import SubarrayDescription


def test_impact_distance():
    """just test a few limit cases. the more detailed tests are done in
    test_shower_impact_distance_off_axis"""

    # test on axis (coming from  zenith onto an array)
    point = np.array([0, 0, 0])
    direction = np.array([0, 0, 1])  # coming perfectly down from zenith
    test_points = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 100], [0, 3, -100]])
    impacts = impact_distance(point=point, direction=direction, test_points=test_points)

    expected_impacts = np.array([0, 1, 2, 3])
    assert np.allclose(impacts, expected_impacts)

    # test at perfectly off-axis (coming in parallel to the array from x-direction)

    direction = np.array([1, 0, 0])  # coming from the x-direction
    test_points = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]]
    )  # points along y axis
    impacts = impact_distance(point=point, direction=direction, test_points=test_points)
    expected_impacts = np.array(
        [0, 1, 2, 3]
    )  # expect the impacts to be equal to the y's
    assert np.allclose(impacts, expected_impacts)

    # test at perfectly off-axis (coming in parallel to the array from y-direction)

    direction = np.array([0, 1, 0])  # coming from the x-direction
    test_points = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]]
    )  # points along y axis
    impacts = impact_distance(point=point, direction=direction, test_points=test_points)
    expected_impacts = np.array(
        [0, 0, 0, 0]
    )  # expect the impacts to be equal to the y's
    assert np.allclose(impacts, expected_impacts)


def test_shower_impact_distance(reference_location):
    """test several boundary cases using the function that takes a Subarray and
    Container
    """

    sub = SubarrayDescription(
        name="test",
        tel_positions={1: [0, 0, 0] * u.m, 2: [0, 1, 0] * u.m, 3: [0, 2, 0] * u.m},
        tel_descriptions={1: None, 2: None, 3: None},
        reference_location=reference_location,
    )

    # coming from zenith to the center of the array, the impact distance should
    # be the cartesian distance
    shower_geom = ReconstructedGeometryContainer(
        core_x=0 * u.m, core_y=0 * u.m, alt=90 * u.deg, az=0 * u.deg
    )
    impact_distances = shower_impact_distance(shower_geom=shower_geom, subarray=sub)
    assert np.allclose(impact_distances, [0, 1, 2] * u.m)

    # alt=0  az=0 should be aligned to x-axis (north in ground frame)
    # therefore the distances should also be just the y-offset of the telescope
    shower_geom = ReconstructedGeometryContainer(
        core_x=0 * u.m, core_y=0 * u.m, alt=0 * u.deg, az=0 * u.deg
    )
    impact_distances = shower_impact_distance(shower_geom=shower_geom, subarray=sub)
    assert np.allclose(impact_distances, [0, 1, 2] * u.m)

    # alt=0  az=90 should be aligned to y-axis (east in ground frame)
    # therefore the distances should also be just the x-offset of the telescope
    shower_geom = ReconstructedGeometryContainer(
        core_x=0 * u.m, core_y=0 * u.m, alt=0 * u.deg, az=90 * u.deg
    )
    impact_distances = shower_impact_distance(shower_geom=shower_geom, subarray=sub)
    assert np.allclose(impact_distances, [0, 0, 0] * u.m)


def test_compare_3d_and_frame_impact_distance(
    example_subarray: SubarrayDescription,
):
    """Test another (slower) way of computing the impact distance, using Frames
    and compare to the implemented method.
    """

    for alt in [90, 0, 50, 60] * u.deg:
        for az in [0, 90, 45, 270, 360] * u.deg:
            shower_geom = ReconstructedGeometryContainer(
                core_x=0 * u.m, core_y=0 * u.m, alt=alt, az=az
            )
            impact_distances_3d = shower_impact_distance(
                shower_geom=shower_geom, subarray=example_subarray
            )

            impact_distances_frame = shower_impact_distance_with_frames(
                shower_geom=shower_geom, subarray=example_subarray
            )

            assert np.allclose(
                impact_distances_frame, impact_distances_3d
            ), f"failed at {alt=} {az=}"
