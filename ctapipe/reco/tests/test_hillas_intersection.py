from ctapipe.reco.hillas_intersection import HillasIntersection
import astropy.units as u
from numpy.testing import assert_allclose
import numpy as np
from astropy.coordinates import SkyCoord
from ctapipe.coordinates import NominalFrame, AltAz, CameraFrame
from ctapipe.containers import HillasParametersContainer

from ctapipe.io import EventSource

from ctapipe.utils import get_dataset_path

from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.calib import CameraCalibrator


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

    array_direction = SkyCoord(
        alt=90 * u.deg - zen_pointing, az=0 * u.deg, frame=horizon_frame
    )
    nom_frame = NominalFrame(origin=array_direction)

    source_sky_pos_reco = SkyCoord(
        alt=90 * u.deg - zen_pointing, az=0 * u.deg, frame=horizon_frame
    )

    nom_pos_reco = source_sky_pos_reco.transform_to(nom_frame)
    delta = 1.0 * u.m

    # LST focal length
    focal_length = 28 * u.m

    hillas_dict = {
        1: HillasParametersContainer(
            x=-(delta / focal_length) * u.rad,
            y=((0 * u.m) / focal_length) * u.rad,
            intensity=1,
        ),
        2: HillasParametersContainer(
            x=((0 * u.m) / focal_length) * u.rad,
            y=-(delta / focal_length) * u.rad,
            intensity=1,
        ),
    }

    x_max = hill_inter.reconstruct_xmax(
        source_x=nom_pos_reco.fov_lon,
        source_y=nom_pos_reco.fov_lat,
        core_x=0 * u.m,
        core_y=0 * u.m,
        hillas_parameters=hillas_dict,
        tel_x={1: (150 * u.m), 2: (0 * u.m)},
        tel_y={1: (0 * u.m), 2: (150 * u.m)},
        zen=zen_pointing,
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
        3: HillasParametersContainer(intensity=100, psi=0 * u.deg),
    }

    reco_konrad = hill_inter.reconstruct_tilted(
        hillas_parameters=hillas_dict, tel_x=tel_x_dict, tel_y=tel_y_dict
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
        3: HillasParametersContainer(intensity=10000, psi=0 * u.deg),
    }

    reco_konrad_spoiled = hill_inter.reconstruct_tilted(
        hillas_parameters=hillas_dict, tel_x=tel_x_dict, tel_y=tel_y_dict
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

    array_direction = SkyCoord(alt=altitude, az=azimuth, frame=horizon_frame)

    nominal_frame = NominalFrame(origin=array_direction)

    focal_length = 28 * u.m

    camera_frame = CameraFrame(
        focal_length=focal_length, telescope_pointing=array_direction
    )

    cog_coords_camera_1 = SkyCoord(x=delta, y=0 * u.m, frame=camera_frame)
    cog_coords_camera_2 = SkyCoord(x=delta / 0.7, y=delta / 0.7, frame=camera_frame)
    cog_coords_camera_3 = SkyCoord(x=0 * u.m, y=delta, frame=camera_frame)

    cog_coords_nom_1 = cog_coords_camera_1.transform_to(nominal_frame)
    cog_coords_nom_2 = cog_coords_camera_2.transform_to(nominal_frame)
    cog_coords_nom_3 = cog_coords_camera_3.transform_to(nominal_frame)

    #  x-axis is along the altitude and y-axis is along the azimuth
    hillas_1 = HillasParametersContainer(
        x=cog_coords_nom_1.fov_lat,
        y=cog_coords_nom_1.fov_lon,
        intensity=100,
        psi=0 * u.deg,
    )

    hillas_2 = HillasParametersContainer(
        x=cog_coords_nom_2.fov_lat,
        y=cog_coords_nom_2.fov_lon,
        intensity=100,
        psi=45 * u.deg,
    )

    hillas_3 = HillasParametersContainer(
        x=cog_coords_nom_3.fov_lat,
        y=cog_coords_nom_3.fov_lon,
        intensity=100,
        psi=90 * u.deg,
    )

    hillas_dict = {1: hillas_1, 2: hillas_2, 3: hillas_3}

    reco_nominal = hill_inter.reconstruct_nominal(hillas_parameters=hillas_dict)

    nominal_pos = SkyCoord(
        fov_lon=u.Quantity(reco_nominal[0], u.rad),
        fov_lat=u.Quantity(reco_nominal[1], u.rad),
        frame=nominal_frame,
    )

    np.testing.assert_allclose(
        nominal_pos.altaz.az.to_value(u.deg), azimuth.to_value(u.deg), atol=1e-8
    )
    np.testing.assert_allclose(
        nominal_pos.altaz.alt.to_value(u.deg), altitude.to_value(u.deg), atol=1e-8
    )


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

    source = EventSource(filename, max_events=10)
    calib = CameraCalibrator(source.subarray)

    horizon_frame = AltAz()

    reconstructed_events = 0

    for event in source:
        calib(event)

        mc = event.simulation.shower
        array_pointing = SkyCoord(az=mc.az, alt=mc.alt, frame=horizon_frame)

        hillas_dict = {}
        telescope_pointings = {}

        for tel_id, dl1 in event.dl1.tel.items():

            geom = source.subarray.tel[tel_id].camera.geometry

            telescope_pointings[tel_id] = SkyCoord(
                alt=event.pointing.tel[tel_id].altitude,
                az=event.pointing.tel[tel_id].azimuth,
                frame=horizon_frame,
            )

            mask = tailcuts_clean(
                geom, dl1.image, picture_thresh=10.0, boundary_thresh=5.0
            )

            try:
                moments = hillas_parameters(geom[mask], dl1.image[mask])
                hillas_dict[tel_id] = moments
            except HillasParameterizationError as e:
                print(e)
                continue

        if len(hillas_dict) < 2:
            continue
        else:
            reconstructed_events += 1

        # divergent mode put to on even though the file has parallel pointing.
        fit_result = fit.predict(
            hillas_dict, source.subarray, array_pointing, telescope_pointings
        )

        print(fit_result)
        print(event.simulation.shower.core_x, event.simulation.shower.core_y)
        fit_result.alt.to(u.deg)
        fit_result.az.to(u.deg)
        fit_result.core_x.to(u.m)
        assert fit_result.is_valid

    assert reconstructed_events > 0


def test_reconstruction_works(subarray_and_event_gamma_off_axis_500_gev):
    subarray, event = subarray_and_event_gamma_off_axis_500_gev
    reconstructor = HillasIntersection()

    array_pointing = SkyCoord(
        az=event.pointing.array_azimuth,
        alt=event.pointing.array_altitude,
        frame=AltAz(),
    )

    hillas_dict = {
        tel_id: dl1.parameters.hillas
        for tel_id, dl1 in event.dl1.tel.items()
        if dl1.parameters.hillas.width.value > 0
    }

    telescope_pointings = {
        tel_id: SkyCoord(alt=pointing.altitude, az=pointing.azimuth, frame=AltAz())
        for tel_id, pointing in event.pointing.tel.items()
        if tel_id in hillas_dict
    }

    result = reconstructor.predict(
        hillas_dict, subarray, array_pointing, telescope_pointings
    )

    reco_coord = SkyCoord(alt=result.alt, az=result.az, frame=AltAz())
    true_coord = SkyCoord(
        alt=event.simulation.shower.alt, az=event.simulation.shower.az, frame=AltAz()
    )

    assert reco_coord.separation(true_coord) < 0.1 * u.deg


def test_selected_subarray(subarray_and_event_gamma_off_axis_500_gev):
    """test that reconstructor also works with "missing" ids"""
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    # remove telescopes 2 and 3 to see that HillasIntersection can work
    # with arbirary telescope ids
    subarray = subarray.select_subarray([1, 4])

    reconstructor = HillasIntersection()
    array_pointing = SkyCoord(
        az=event.pointing.array_azimuth,
        alt=event.pointing.array_altitude,
        frame=AltAz(),
    )

    # again, only use telescopes 1 and 4
    hillas_dict = {
        tel_id: dl1.parameters.hillas
        for tel_id, dl1 in event.dl1.tel.items()
        if dl1.parameters.hillas.width.value > 0 and tel_id in {1, 4}
    }

    telescope_pointings = {
        tel_id: SkyCoord(alt=pointing.altitude, az=pointing.azimuth, frame=AltAz())
        for tel_id, pointing in event.pointing.tel.items()
        if tel_id in hillas_dict
    }

    reconstructor.predict(hillas_dict, subarray, array_pointing, telescope_pointings)
