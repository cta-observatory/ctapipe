import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from numpy.testing import assert_allclose

from ctapipe.containers import HillasParametersContainer
from ctapipe.coordinates import CameraFrame, NominalFrame
from ctapipe.instrument import SubarrayDescription
from ctapipe.reco.hillas_intersection import HillasIntersection


def test_intersect():
    """
    Simple test to check the intersection of lines. Try to intersect positions at (0,1) and (1,0)
    with angles perpendicular and test they cross at (0,0)
    """
    x1 = 0
    y1 = 1
    theta1 = 90 * u.deg

    x2 = 1
    y2 = 0
    theta2 = 0 * u.deg

    sx, sy = HillasIntersection.intersect_lines(x1, y1, theta1, x2, y2, theta2)

    assert_allclose(sx, 0, atol=1e-6)
    assert_allclose(sy, 0, atol=1e-6)


def test_parallel():
    """
    ?    Simple test to check the intersection of lines. Try to intersect positions at (0,0) and (0,1)
        with angles parallel and check the behaviour
    """
    x1 = 0
    y1 = 0
    theta1 = 0 * u.deg

    x2 = 1
    y2 = 0
    theta2 = 0 * u.deg

    sx, sy = HillasIntersection.intersect_lines(x1, y1, theta1, x2, y2, theta2)
    assert_allclose(sx, np.nan, atol=1e-6)
    assert_allclose(sy, np.nan, atol=1e-6)


def test_intersection_xmax_reco(example_subarray):
    """
    Test the reconstruction of xmax with two LSTs that are pointing at zenith = 0.
    The telescopes are places along the x and y axis at the same distance from the center.
    The impact point is hard-coded to be happening in the center of this cartesian system.
    """
    hill_inter = HillasIntersection(subarray=example_subarray)

    altaz = AltAz()
    zen_pointing = 10 * u.deg

    array_direction = SkyCoord(alt=90 * u.deg - zen_pointing, az=0 * u.deg, frame=altaz)
    nom_frame = NominalFrame(origin=array_direction)

    source_sky_pos_reco = SkyCoord(
        alt=90 * u.deg - zen_pointing, az=0 * u.deg, frame=altaz
    )

    nom_pos_reco = source_sky_pos_reco.transform_to(nom_frame)
    delta = 1.0 * u.m

    # LST focal length
    focal_length = 28 * u.m

    hillas_dict = {
        1: HillasParametersContainer(
            fov_lon=-(delta / focal_length) * u.rad,
            fov_lat=((0 * u.m) / focal_length) * u.rad,
            intensity=1,
        ),
        2: HillasParametersContainer(
            fov_lon=((0 * u.m) / focal_length) * u.rad,
            fov_lat=-(delta / focal_length) * u.rad,
            intensity=1,
        ),
    }

    h_max = hill_inter.reconstruct_h_max(
        source_x=nom_pos_reco.fov_lon,
        source_y=nom_pos_reco.fov_lat,
        core_x=0 * u.m,
        core_y=0 * u.m,
        hillas_parameters=hillas_dict,
        tel_x={1: (150 * u.m), 2: (0 * u.m)},
        tel_y={1: (0 * u.m), 2: (150 * u.m)},
        zen=zen_pointing,
    )

    assert h_max > 0 * u.m


def test_intersection_reco_impact_point_tilted(example_subarray):
    """
    Function to test the reconstruction of the impact point in the tilted frame.
    This is done using a squared configuration, of which the impact point occupies a vertex,
    ad the three telescopes the other three vertices.
    """
    hill_inter = HillasIntersection(example_subarray)

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

    np.testing.assert_allclose(reco_konrad[0], -delta.to_value(u.m), atol=1e-6)
    np.testing.assert_allclose(reco_konrad[1], delta.to_value(u.m), atol=1e-6)


def test_intersection_weighting_spoiled_parameters(example_subarray):
    """
    Test that the weighting scheme is useful especially when a telescope is 90 deg with respect to the other two
    """
    hill_inter = HillasIntersection(example_subarray)

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

    np.testing.assert_allclose(reco_konrad_spoiled[0], -delta.to_value(u.m), atol=1e-1)
    np.testing.assert_allclose(reco_konrad_spoiled[1], delta.to_value(u.m), atol=1e-1)


def test_intersection_nominal_reconstruction(example_subarray):
    """
    Testing the reconstruction of the position in the nominal frame with a three-telescopes system.
    This is done using a squared configuration, of which the impact point occupies a vertex,
    ad the three telescopes the other three vertices.
    """
    hill_inter = HillasIntersection(example_subarray)

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

    cog_coords_camera_1 = SkyCoord(y=delta, x=0 * u.m, frame=camera_frame)
    cog_coords_camera_2 = SkyCoord(y=delta / 0.7, x=delta / 0.7, frame=camera_frame)
    cog_coords_camera_3 = SkyCoord(y=0 * u.m, x=delta, frame=camera_frame)

    cog_coords_nom_1 = cog_coords_camera_1.transform_to(nominal_frame)
    cog_coords_nom_2 = cog_coords_camera_2.transform_to(nominal_frame)
    cog_coords_nom_3 = cog_coords_camera_3.transform_to(nominal_frame)

    hillas_1 = HillasParametersContainer(
        fov_lat=cog_coords_nom_1.fov_lat,
        fov_lon=cog_coords_nom_1.fov_lon,
        intensity=100,
        psi=0 * u.deg,
    )

    hillas_2 = HillasParametersContainer(
        fov_lat=cog_coords_nom_2.fov_lat,
        fov_lon=cog_coords_nom_2.fov_lon,
        intensity=100,
        psi=45 * u.deg,
    )

    hillas_3 = HillasParametersContainer(
        fov_lat=cog_coords_nom_3.fov_lat,
        fov_lon=cog_coords_nom_3.fov_lon,
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


def test_badly_reconstructed_event(prod5_mst_flashcam):
    """
    Test that events reconstructed at large angular distance
    from FoV center return INVALID. Event and array loosely follow an
    actual simulation event.
    """
    tel_pos = {1: np.array([150, 75, 0]) * u.m, 2: np.array([150, -240, 0]) * u.m}
    tel_desc = {1: prod5_mst_flashcam, 2: prod5_mst_flashcam}
    reference_location = EarthLocation(
        lon=-70.32 * u.deg,
        lat=-24.68 * u.deg,
        height=2147 * u.m,
    )
    two_tel_subarray = SubarrayDescription(
        "two_tel_subarray",
        tel_positions=tel_pos,
        tel_descriptions=tel_desc,
        reference_location=reference_location,
    )
    hill_inter = HillasIntersection(two_tel_subarray)

    hillas_tel_1 = HillasParametersContainer(
        fov_lon=0.6 * u.deg,
        fov_lat=-0.3 * u.deg,
        intensity=95,
        psi=85.3 * u.deg,
        length=0.09 * u.deg,
        width=0.05 * u.deg,
    )

    hillas_tel_2 = HillasParametersContainer(
        fov_lon=-0.1 * u.deg,
        fov_lat=-0.4 * u.deg,
        intensity=119,
        psi=85.0 * u.deg,
        length=0.15 * u.deg,
        width=0.04 * u.deg,
    )

    hillas_dir = {1: hillas_tel_1, 2: hillas_tel_2}
    pointing = AltAz(alt=70 * u.deg, az=0 * u.deg)
    reco_event = hill_inter._predict(hillas_dict=hillas_dir, array_pointing=pointing)

    assert not reco_event.is_valid


def test_reconstruction_works(subarray_and_event_gamma_off_axis_500_gev):
    subarray, event = subarray_and_event_gamma_off_axis_500_gev
    reconstructor = HillasIntersection(subarray)

    true_coord = SkyCoord(
        alt=event.simulation.shower.alt, az=event.simulation.shower.az, frame=AltAz()
    )

    reconstructor(event)
    result = event.dl2.stereo.geometry["HillasIntersection"]
    reco_coord = SkyCoord(alt=result.alt, az=result.az, frame=AltAz())
    assert reco_coord.separation(true_coord) < 0.1 * u.deg


def test_selected_subarray(subarray_and_event_gamma_off_axis_500_gev):
    """test that reconstructor also works with "missing" ids"""
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    # remove telescopes 2 and 3 to see that HillasIntersection can work
    # with arbitrary telescope ids
    allowed_tels = {1, 4}
    for tel_id in subarray.tel.keys():
        if tel_id not in allowed_tels:
            event.dl1.tel.pop(tel_id, None)
            event.trigger.tels_with_trigger = event.trigger.tels_with_trigger[
                event.trigger.tels_with_trigger != tel_id
            ]

    subarray = subarray.select_subarray(allowed_tels)

    reconstructor = HillasIntersection(subarray)
    reconstructor(event)
    result = event.dl2.stereo.geometry["HillasIntersection"]
    assert result.is_valid
