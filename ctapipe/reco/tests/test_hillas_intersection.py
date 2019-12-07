from ctapipe.reco.hillas_intersection import HillasIntersection
import astropy.units as u
from numpy.testing import assert_allclose
import numpy as np
from astropy.coordinates import SkyCoord
from ctapipe.coordinates import NominalFrame, AltAz, CameraFrame
from ctapipe.io.containers import HillasParametersContainer

from ctapipe.io import event_source

from ctapipe.utils import get_dataset_path

from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

import pytest

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
    zen_pointing = 10 * u.deg

    array_direction = SkyCoord(alt=90*u.deg - zen_pointing,
                               az=0 * u.deg,
                               frame=horizon_frame)
    nom_frame = NominalFrame(origin=array_direction)

    source_sky_pos_reco = SkyCoord(alt=90 * u.deg - zen_pointing,
                                   az=0 * u.deg,
                                   frame=horizon_frame)

    nom_pos_reco = source_sky_pos_reco.transform_to(nom_frame)
    delta = 1.0 * u.m

    # LST focal length
    focal_length = 28 * u.m

    hillas_dict = {
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
        hillas_parameters=hillas_dict,
        tel_x={1: (150 * u.m), 2: (0 * u.m)},
        tel_y={1: (0 * u.m), 2: (150 * u.m)},
        zen=zen_pointing
    )
    print(x_max)


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


def test_intersection_nominal_reconstruction():
    """
    Testing the reconstruction of the position in the nominal frame with a three-telescopes system.
    This is done using a squared configuration, of which the impact point occupies a vertex,
    ad the three telescopes the other three vertices.
    """
    hill_inter = HillasIntersection()

    delta = 1.0 * u.m
    horizon_frame = AltAz()
    altitude = 70 * u.deg
    azimuth = 10 * u.deg

    array_direction = SkyCoord(alt=altitude,
                               az=azimuth,
                               frame=horizon_frame)

    nominal_frame = NominalFrame(origin=array_direction)

    focal_length = 28 * u.m

    camera_frame = CameraFrame(focal_length=focal_length,
                               telescope_pointing=array_direction)

    cog_coords_camera_1 = SkyCoord(x=delta, y=0 * u.m, frame=camera_frame)
    cog_coords_camera_2 = SkyCoord(x=delta / 0.7, y=delta / 0.7, frame=camera_frame)
    cog_coords_camera_3 = SkyCoord(x=0 * u.m, y=delta, frame=camera_frame)

    cog_coords_nom_1 = cog_coords_camera_1.transform_to(nominal_frame)
    cog_coords_nom_2 = cog_coords_camera_2.transform_to(nominal_frame)
    cog_coords_nom_3 = cog_coords_camera_3.transform_to(nominal_frame)

    #  x-axis is along the altitude and y-axis is along the azimuth
    hillas_1 = HillasParametersContainer(x=cog_coords_nom_1.delta_alt,
                                         y=cog_coords_nom_1.delta_az,
                                         intensity=100,
                                         psi=0 * u.deg)

    hillas_2 = HillasParametersContainer(x=cog_coords_nom_2.delta_alt,
                                         y=cog_coords_nom_2.delta_az,
                                         intensity=100,
                                         psi=45 * u.deg)

    hillas_3 = HillasParametersContainer(x=cog_coords_nom_3.delta_alt,
                                         y=cog_coords_nom_3.delta_az,
                                         intensity=100,
                                         psi=90 * u.deg)

    hillas_dict = {1: hillas_1, 2: hillas_2, 3: hillas_3}

    reco_nominal = hill_inter.reconstruct_nominal(hillas_parameters=hillas_dict)

    nominal_pos = SkyCoord(
        delta_az=u.Quantity(reco_nominal[0], u.rad),
        delta_alt=u.Quantity(reco_nominal[1], u.rad),
        frame=nominal_frame
    )

    np.testing.assert_allclose(nominal_pos.altaz.az.to_value(u.deg), azimuth.to_value(u.deg), atol=1e-8)
    np.testing.assert_allclose(nominal_pos.altaz.alt.to_value(u.deg), altitude.to_value(u.deg), atol=1e-8)


def test_reconstruction():
    """
    a test of the complete fit procedure on one event including:
    • tailcut cleaning
    • hillas parametrisation
    • direction fit
    • position fit

    in the end, proper units in the output are asserted """

    filename = get_dataset_path("gamma_test_large.simtel.gz")

    fit = HillasIntersection()

    source = event_source(filename, max_events=10)

    horizon_frame = AltAz()

    reconstructed_events = 0

    for event in source:
        array_pointing = SkyCoord(
            az=event.mc.az,
            alt=event.mc.alt,
            frame=horizon_frame
        )

        hillas_dict = {}
        telescope_pointings = {}

        for tel_id in event.dl0.tels_with_data:

            geom = event.inst.subarray.tel[tel_id].camera

            telescope_pointings[tel_id] = SkyCoord(alt=event.pointing[tel_id].altitude,
                                                   az=event.pointing[tel_id].azimuth,
                                                   frame=horizon_frame)
            pmt_signal = event.r0.tel[tel_id].waveform[0].sum(axis=1)

            mask = tailcuts_clean(geom, pmt_signal,
                                  picture_thresh=10., boundary_thresh=5.)
            pmt_signal[mask == 0] = 0

            try:
                moments = hillas_parameters(geom, pmt_signal)
                hillas_dict[tel_id] = moments
            except HillasParameterizationError as e:
                print(e)
                continue

        if len(hillas_dict) < 2:
            continue
        else:
            reconstructed_events += 1

        # divergent mode put to on even though the file has parallel pointing.
        fit_result = fit.predict(hillas_dict, event.inst, array_pointing, telescope_pointings)

        print(fit_result)
        print(event.mc.core_x, event.mc.core_y)
        fit_result.alt.to(u.deg)
        fit_result.az.to(u.deg)
        fit_result.core_x.to(u.m)
        assert fit_result.is_valid

    assert reconstructed_events > 0
