from copy import deepcopy
import numpy as np
from astropy import units as u
import pytest

from ctapipe.containers import ImageParametersContainer, HillasParametersContainer
from ctapipe.instrument import SubarrayDescription, TelescopeDescription
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.io import SimTelEventSource
from ctapipe.reco.hillas_reconstructor import HillasReconstructor, HillasPlane
from ctapipe.utils import get_dataset_path
from ctapipe.coordinates import TelescopeFrame
from astropy.coordinates import SkyCoord, AltAz
from ctapipe.calib import CameraCalibrator


def test_estimator_results():
    """
    creating some planes pointing in different directions (two
    north-south, two east-west) and that have a slight position errors (+-
    0.1 m in one of the four cardinal directions """
    horizon_frame = AltAz()

    p1 = SkyCoord(alt=43 * u.deg, az=45 * u.deg, frame=horizon_frame)
    p2 = SkyCoord(alt=47 * u.deg, az=45 * u.deg, frame=horizon_frame)
    circle1 = HillasPlane(p1=p1, p2=p2, telescope_position=[0, 1, 0] * u.m)

    p1 = SkyCoord(alt=44 * u.deg, az=90 * u.deg, frame=horizon_frame)
    p2 = SkyCoord(alt=46 * u.deg, az=90 * u.deg, frame=horizon_frame)
    circle2 = HillasPlane(p1=p1, p2=p2, telescope_position=[1, 0, 0] * u.m)

    p1 = SkyCoord(alt=44.5 * u.deg, az=45 * u.deg, frame=horizon_frame)
    p2 = SkyCoord(alt=46.5 * u.deg, az=45 * u.deg, frame=horizon_frame)
    circle3 = HillasPlane(p1=p1, p2=p2, telescope_position=[0, -1, 0] * u.m)

    p1 = SkyCoord(alt=43.5 * u.deg, az=90 * u.deg, frame=horizon_frame)
    p2 = SkyCoord(alt=45.5 * u.deg, az=90 * u.deg, frame=horizon_frame)
    circle4 = HillasPlane(p1=p1, p2=p2, telescope_position=[-1, 0, 0] * u.m)

    # Create a dummy subarray
    # (not used here, but required to initialize the reconstructor)
    subarray = SubarrayDescription(
        "test array",
        tel_positions={1: np.zeros(3) * u.m},
        tel_descriptions={
            1: TelescopeDescription.from_name(
                optics_name="SST-ASTRI", camera_name="CHEC"
            )
        },
    )

    # creating the fit class and setting the the great circle member
    fit = HillasReconstructor(subarray)
    hillas_planes = {1: circle1, 2: circle2, 3: circle3, 4: circle4}

    # performing the direction fit with the minimisation algorithm
    # and a seed that is perpendicular to the up direction
    dir_fit_minimise, _ = fit.estimate_direction(hillas_planes)
    print("direction fit test minimise:", dir_fit_minimise)


def test_h_max_results(example_subarray):
    """
    creating some planes pointing in different directions (two
    north-south, two east-west) and that have a slight position errors (+-
    0.1 m in one of the four cardinal directions """
    horizon_frame = AltAz()

    p1 = SkyCoord(alt=0 * u.deg, az=45 * u.deg, frame=horizon_frame)
    p2 = SkyCoord(alt=0 * u.deg, az=45 * u.deg, frame=horizon_frame)
    circle1 = HillasPlane(p1=p1, p2=p2, telescope_position=[0, 1, 0] * u.m)

    p1 = SkyCoord(alt=0 * u.deg, az=90 * u.deg, frame=horizon_frame)
    p2 = SkyCoord(alt=0 * u.deg, az=90 * u.deg, frame=horizon_frame)
    circle2 = HillasPlane(p1=p1, p2=p2, telescope_position=[1, 0, 0] * u.m)

    p1 = SkyCoord(alt=0 * u.deg, az=45 * u.deg, frame=horizon_frame)
    p2 = SkyCoord(alt=0 * u.deg, az=45 * u.deg, frame=horizon_frame)
    circle3 = HillasPlane(p1=p1, p2=p2, telescope_position=[0, -1, 0] * u.m)

    p1 = SkyCoord(alt=0 * u.deg, az=90 * u.deg, frame=horizon_frame)
    p2 = SkyCoord(alt=0 * u.deg, az=90 * u.deg, frame=horizon_frame)
    circle4 = HillasPlane(p1=p1, p2=p2, telescope_position=[-1, 0, 0] * u.m)

    # Create a dummy subarray
    # (not used here, but required to initialize the reconstructor)
    subarray = example_subarray

    # creating the fit class and setting the the great circle member
    fit = HillasReconstructor(subarray)
    hillas_planes = {1: circle1, 2: circle2, 3: circle3, 4: circle4}

    # performing the direction fit with the minimisation algorithm
    # and a seed that is perpendicular to the up direction
    h_max_reco = fit.estimate_h_max(hillas_planes)
    print("h max fit test minimise:", h_max_reco)

    # the results should be close to the direction straight up
    np.testing.assert_allclose(h_max_reco.value, 0, atol=1e-8)
    # np.testing.assert_allclose(fitted_core_position.value, [0, 0], atol=1e-3)


def test_invalid_events(subarray_and_event_gamma_off_axis_500_gev):
    """
    The HillasReconstructor is supposed to fail
    in these cases:
    - less than two teleskopes
    - any width is NaN
    - any width is 0

    This test takes 1 shower from a test simtel file and modifies a-posteriori
    some hillas dictionaries to make it non-reconstructable.
    It is supposed to fail if no Exception or another Exception gets thrown.
    """

    # 4-LST bright event already calibrated
    # we'll clean it and parametrize it again in the TelescopeFrame
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    tel_azimuth = {}
    tel_altitude = {}

    #source = EventSource(filename, max_events=1)
    #subarray = source.subarray
    calib = CameraCalibrator(subarray)
    fit = HillasReconstructor(subarray)

    #for event in source:

    calib(event)

    hillas_dict = {}
    for tel_id, dl1 in event.dl1.tel.items():

        geom = subarray.tel[tel_id].camera.geometry
        tel_azimuth[tel_id] = event.pointing.tel[tel_id].azimuth
        tel_altitude[tel_id] = event.pointing.tel[tel_id].altitude

        mask = tailcuts_clean(
            geom, dl1.image, picture_thresh=10.0, boundary_thresh=5.0
        )

        dl1.parameters = ImageParametersContainer()

        try:
            moments = hillas_parameters(geom[mask], dl1.image[mask])
            hillas_dict[tel_id] = moments
            dl1.parameters.hillas = moments
        except HillasParameterizationError:
            dl1.parameters.hillas = HillasParametersContainer()
            continue

    # copy event container to modify it
    event_copy = deepcopy(event)
    # overwrite all image parameters but the last one with dummy ones
    for tel_id in list(event_copy.dl1.tel.keys())[:-1]:
        event_copy.dl1.tel[tel_id].parameters.hillas = HillasParametersContainer()
    fit(event_copy)
    assert event_copy.dl2.stereo.geometry["HillasReconstructor"].is_valid is False

    # Now use the original event, but overwrite the last width to 0
    event.dl1.tel[tel_id].parameters.hillas.width = 0 * u.m
    fit(event)
    assert event.dl2.stereo.geometry["HillasReconstructor"].is_valid is False

    # Now use the original event, but overwrite the last width to NaN
    event.dl1.tel[tel_id].parameters.hillas.width = np.nan * u.m
    fit(event)
    assert event.dl2.stereo.geometry["HillasReconstructor"].is_valid is False


def test_reconstruction_against_simulation(subarray_and_event_gamma_off_axis_500_gev):
    """Reconstruction is here done only in the TelescopeFrame,
    since the previous tests test already for the compatibility between
    frames"""

    # 4-LST bright event already calibrated
    # we'll clean it and parametrize it again in the TelescopeFrame
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    # define reconstructor
    reconstructor = HillasReconstructor(subarray)

    hillas_dict = {}
    telescope_pointings = {}

    for tel_id, dl1 in event.dl1.tel.items():

        telescope_pointings[tel_id] = SkyCoord(
            alt=event.pointing.tel[tel_id].altitude,
            az=event.pointing.tel[tel_id].azimuth,
            frame=AltAz(),
        )

        geom_CameraFrame = subarray.tel[tel_id].camera.geometry

        # this could be done also out of this loop,
        # but in case of real data each telescope would have a
        # different telescope_pointing
        geom_TelescopeFrame = geom_CameraFrame.transform_to(
            TelescopeFrame(telescope_pointing=telescope_pointings[tel_id])
        )

        mask = tailcuts_clean(
            geom_TelescopeFrame,
            dl1.image,
            picture_thresh=5.0,
            boundary_thresh=2.5,
            keep_isolated_pixels=False,
            min_number_picture_neighbors=2,
        )

        try:
            hillas_dict[tel_id] = hillas_parameters(
                geom_TelescopeFrame[mask], dl1.image[mask]
            )

            # the original event is created from a
            # pytest fixture with "session" scope, so it's always the same
            # and if we used the same event we would overwrite the image
            # parameters for the next tests, thus causing their failure
            test_event = deepcopy(event)
            test_event.dl1.tel[tel_id].parameters = ImageParametersContainer()
            test_event.dl1.tel[tel_id].parameters.hillas = hillas_dict[tel_id]

        except HillasParameterizationError as e:
            print(e)
            continue

    # Get shower geometry
    reconstructor(event)
    # get the result from the correct DL2 container
    result = event.dl2.stereo.geometry["HillasReconstructor"]

    # get the reconstructed coordinates in the sky
    reco_coord = SkyCoord(alt=result.alt, az=result.az, frame=AltAz())
    # get the simulated coordinates in the sky
    true_coord = SkyCoord(
        alt=event.simulation.shower.alt, az=event.simulation.shower.az, frame=AltAz()
    )

    # check that we are not more far than 0.1 degrees
    assert reco_coord.separation(true_coord) < 0.1 * u.deg


@pytest.mark.parametrize("filename", 
                         ["gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz",
                         "gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz"])
def test_CameraFrame_against_TelescopeFrame(filename):

    input_file = get_dataset_path(
        "gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz"
    )

    source = SimTelEventSource(input_file, max_events=10)

    calib = CameraCalibrator(subarray=source.subarray)
    reconstructor = HillasReconstructor(source.subarray)

    reconstructed_events = 0

    for event in source:

        calib(event)
        # make a copy of the calibrated event for the camera frame case
        # later we clean and paramretrize the 2 events in the same way
        # but in 2 different frames to check they return compatible results
        event_camera_frame = deepcopy(event)

        telescope_pointings = {}
        hillas_dict_camera_frame = {}
        hillas_dict_telescope_frame = {}

        for tel_id, dl1 in event.dl1.tel.items():

            event_camera_frame.dl1.tel[tel_id].parameters = ImageParametersContainer()
            event.dl1.tel[tel_id].parameters = ImageParametersContainer()

            # this is needed only here to transform the camera geometries
            telescope_pointings[tel_id] = SkyCoord(
                alt=event.pointing.tel[tel_id].altitude,
                az=event.pointing.tel[tel_id].azimuth,
                frame=AltAz(),
            )

            geom_camera_frame = source.subarray.tel[tel_id].camera.geometry

            # this could be done also out of this loop,
            # but in case of real data each telescope would have a
            # different telescope_pointing
            geom_telescope_frame = geom_camera_frame.transform_to(
                TelescopeFrame(telescope_pointing=telescope_pointings[tel_id])
            )

            mask = tailcuts_clean(
                geom_telescope_frame, dl1.image, picture_thresh=10.0, boundary_thresh=5.0
            )

            try:
                moments_camera_frame = hillas_parameters(
                    geom_camera_frame[mask], dl1.image[mask]
                )
                moments_telescope_frame = hillas_parameters(
                    geom_telescope_frame[mask], dl1.image[mask]
                )

                if (moments_camera_frame.width.value > 0) and (moments_telescope_frame.width.value > 0):
                    event_camera_frame.dl1.tel[
                        tel_id
                    ].parameters.hillas = moments_camera_frame
                    dl1.parameters.hillas = moments_telescope_frame

                    hillas_dict_camera_frame[tel_id] = moments_camera_frame
                    hillas_dict_telescope_frame[tel_id] = moments_telescope_frame
                else:
                    continue

            except HillasParameterizationError as e:
                print(e)
                continue

        if (len(hillas_dict_camera_frame) > 2) and (len(hillas_dict_telescope_frame) > 2):
            reconstructor(event_camera_frame)
            reconstructor(event)
            reconstructed_events += 1
        else:  # this event was not good enough to be tested on
            continue

        # Compare old approach with new approach
        result_camera_frame = event_camera_frame.dl2.stereo.geometry["HillasReconstructor"]
        result_telescope_frame = event.dl2.stereo.geometry["HillasReconstructor"]

        assert result_camera_frame.is_valid
        assert result_telescope_frame.is_valid

        for field in event.dl2.stereo.geometry["HillasReconstructor"].as_dict():
            C = np.asarray(result_camera_frame.as_dict()[field])
            T = np.asarray(result_telescope_frame.as_dict()[field])
            assert (np.isclose(C, T, rtol=1e-03, atol=1e-03, equal_nan=True)).all()

    assert reconstructed_events > 0 # check that we reconstruct at least 1 event