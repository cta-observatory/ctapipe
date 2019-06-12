from ctapipe.reco.hillas_intersection import HillasIntersection
import astropy.units as u
from numpy.testing import assert_allclose
import numpy as np
from astropy.coordinates import SkyCoord
from ctapipe.coordinates import NominalFrame, AltAz
from ctapipe.io.containers import HillasParametersContainer


def test_intersect():
    """
    Simple test to check the intersection of lines. Try to intersect positions at (0,1) and (1,0)
    with angles perpendicular and test they cross at (0,0)
    """
    hill = HillasIntersection()
    x1 = 0
    y1 = 1
    theta1 = 90 * u.deg

    x2 = 1
    y2 = 0
    theta2 = 0 * u.deg

    sx, sy = hill.intersect_lines(x1, y1, theta1, x2, y2, theta2)

    assert_allclose(sx, 0, atol=1e-6)
    assert_allclose(sy, 0, atol=1e-6)


def test_parallel():
    """
    Simple test to check the intersection of lines. Try to intersect positions at (0,0) and (0,1)
    with angles parallel and check the behaviour
    """
    hill = HillasIntersection()
    x1 = 0
    y1 = 0
    theta1 = 0 * u.deg

    x2 = 1
    y2 = 0
    theta2 = 0 * u.deg

    sx, sy = hill.intersect_lines(x1, y1, theta1, x2, y2, theta2)
    assert_allclose(sx, np.nan, atol=1e-6)
    assert_allclose(sy, np.nan, atol=1e-6)


def test_intersection_xmax_reco():
    """
    Test the reconstruction of xmax with two LSTs that are pointing at zenith = 0.
    The telescopes are places along the x and y axis at the same distance from the center.
    The impact point is hard-coded to be happening in the center of this cartesian system.
    """
    hill_inter = HillasIntersection()

    horizon_frame = AltAz()
    zen_pointing = 0 * u.deg
    array_direction = SkyCoord(alt=90*u.deg,
                               az=0 * u.deg,
                               frame=horizon_frame)
    nom_frame = NominalFrame(origin=array_direction)

    source_sky_pos_reco = SkyCoord(alt=90 * u.deg,
                                   az=0 * u.deg,
                                   frame=horizon_frame)

    nom_pos_reco = source_sky_pos_reco.transform_to(nom_frame)
    delta = 1.0 * u.m

    # LST focal length
    focal_length = 28 * u.m

    hillas_parameters = {
        1: HillasParametersContainer(x=-(delta/focal_length)*u.rad,
                                     y=((0 * u.m)/focal_length) * u.rad,
                                     intensity=1),
        2: HillasParametersContainer(x=((0 * u.m)/focal_length) * u.rad,
                                     y=-(delta/focal_length) * u.rad,
                                     intensity=1)
    }

    x_max = hill_inter.reconstruct_xmax(
        source_x=nom_pos_reco.delta_az,
        source_y=nom_pos_reco.delta_alt,
        core_x=0 * u.m,
        core_y=0 * u.m,
        hillas_parameters=hillas_parameters,
        tel_x={1: (150 * u.m), 2: (0 * u.m)},
        tel_y={1: (0 * u.m), 2: (150 * u.m)},
        zen=zen_pointing
    )
    # TODO: ADD A PROPER ASSERT


def test_intersection_reco_impact_point_tilted():
    """
    Function to test the reconstruction of the impact point in the tilted frame.
    This is done using a squared configuration, of which the impact point occupies a vertex,
    ad the three telescopes the other three vertices.
    """
    hill_inter = HillasIntersection()

    delta = 100 * u.m
    tel_x_dict = {1: delta, 2: -delta, 3: -delta}
    tel_y_dict = {1: delta, 2: delta, 3: -delta}

    hillas_dict = {
        1: HillasParametersContainer(intensity=100, psi=-90 * u.deg),
        2: HillasParametersContainer(intensity=100, psi=-45 * u.deg),
        3: HillasParametersContainer(intensity=100, psi=0 * u.deg)
    }

    reco_konrad = hill_inter.reconstruct_tilted(
        hillas_parameters=hillas_dict,
        tel_x=tel_x_dict,
        tel_y=tel_y_dict
    )

    np.testing.assert_allclose(reco_konrad[0], delta.to_value(u.m), atol=1e-8)
    np.testing.assert_allclose(reco_konrad[1], -delta.to_value(u.m), atol=1e-8)


def test_intersection_weighting_spoiled_parameters():
    """
    Test that the weighting scheme is useful especially when a telescope is 90 deg with respect to the other two
    """
    hill_inter = HillasIntersection()

    delta = 100 * u.m
    tel_x_dict = {1: delta, 2: -delta, 3: -delta}
    tel_y_dict = {1: delta, 2: delta, 3: -delta}

    # telescope 2 have a spoiled reconstruction (45 instead of -45)
    hillas_dict = {
        1: HillasParametersContainer(intensity=10000, psi=-90 * u.deg),
        2: HillasParametersContainer(intensity=1, psi=45 * u.deg),
        3: HillasParametersContainer(intensity=10000, psi=0 * u.deg)
    }

    reco_konrad_spoiled = hill_inter.reconstruct_tilted(
        hillas_parameters=hillas_dict,
        tel_x=tel_x_dict,
        tel_y=tel_y_dict
    )

    np.testing.assert_allclose(reco_konrad_spoiled[0], delta.to_value(u.m), atol=1e-1)
    np.testing.assert_allclose(reco_konrad_spoiled[1], -delta.to_value(u.m), atol=1e-1)
