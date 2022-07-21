from copy import deepcopy

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from traitlets.config import Config

from ctapipe.calib import CameraCalibrator
from ctapipe.containers import HillasParametersContainer
from ctapipe.image.image_processor import ImageProcessor
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import SimTelEventSource
from ctapipe.reco.hillas_reconstructor import HillasPlane, HillasReconstructor
from ctapipe.utils import get_dataset_path


def test_estimator_results(prod5_sst):
    """
    creating some planes pointing in different directions (two
    north-south, two east-west) and that have a slight position errors (+-
    0.1 m in one of the four cardinal directions"""
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
        tel_descriptions={1: prod5_sst},
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
    0.1 m in one of the four cardinal directions"""
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
    calib = CameraCalibrator(subarray)
    image_processor = ImageProcessor(subarray)

    # perform no quality cuts, so we can see if our additional checks on valid
    # input work
    config = Config(
        {
            "StereoQualityQuery": {
                "quality_criteria": [],
            }
        }
    )
    hillas_reconstructor = HillasReconstructor(subarray, config=config)

    calib(event)
    image_processor(event)
    original_event = deepcopy(event)

    hillas_reconstructor(event)
    result = event.dl2.stereo.geometry["HillasReconstructor"]
    assert result.is_valid

    # copy event container to modify it
    invalid_event = deepcopy(original_event)

    # overwrite all image parameters but the last one with dummy ones
    for tel_id in list(invalid_event.dl1.tel.keys())[:-1]:
        invalid_event.dl1.tel[tel_id].parameters.hillas = HillasParametersContainer()

    hillas_reconstructor(invalid_event)
    result = invalid_event.dl2.stereo.geometry["HillasReconstructor"]
    assert not result.is_valid

    tel_id = list(invalid_event.dl1.tel.keys())[-1]
    # Now use the original event, but overwrite the last width to 0
    invalid_event = deepcopy(original_event)
    invalid_event.dl1.tel[tel_id].parameters.hillas.width = 0 * u.m
    hillas_reconstructor(invalid_event)
    result = invalid_event.dl2.stereo.geometry["HillasReconstructor"]
    assert not result.is_valid

    # Now use the original event, but overwrite the last width to NaN
    invalid_event = deepcopy(original_event)
    invalid_event.dl1.tel[tel_id].parameters.hillas.width = np.nan * u.m
    hillas_reconstructor(invalid_event)
    result = invalid_event.dl2.stereo.geometry["HillasReconstructor"]
    assert not result.is_valid


def test_reconstruction_against_simulation(subarray_and_event_gamma_off_axis_500_gev):
    """Reconstruction is here done only in the TelescopeFrame,
    since the previous tests test already for the compatibility between
    frames"""

    # 4-LST bright event already calibrated
    # we'll clean it and parametrize it again in the TelescopeFrame
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    # define reconstructor
    calib = CameraCalibrator(subarray)
    image_processor = ImageProcessor(subarray)
    reconstructor = HillasReconstructor(subarray)

    # Get shower geometry
    calib(event)
    image_processor(event)
    reconstructor(event)
    result = event.dl2.stereo.geometry["HillasReconstructor"]

    # get the reconstructed coordinates in the sky
    reco_coord = SkyCoord(alt=result.alt, az=result.az, frame=AltAz())
    # get the simulated coordinates in the sky
    true_coord = SkyCoord(
        alt=event.simulation.shower.alt, az=event.simulation.shower.az, frame=AltAz()
    )

    # check that we are not more far than 0.1 degrees
    assert reco_coord.separation(true_coord) < 0.1 * u.deg

    assert u.isclose(result.core_x, event.simulation.shower.core_x, atol=25 * u.m)
    assert u.isclose(result.core_y, event.simulation.shower.core_y, atol=25 * u.m)


def test_reconstruction_against_simulation_camera_frame(
    subarray_and_event_gamma_off_axis_500_gev,
):
    """Reconstruction is here done only in the TelescopeFrame,
    since the previous tests test already for the compatibility between
    frames"""

    # 4-LST bright event already calibrated
    # we'll clean it and parametrize it again in the TelescopeFrame
    subarray, event = subarray_and_event_gamma_off_axis_500_gev

    # define reconstructor
    calib = CameraCalibrator(subarray)
    image_processor = ImageProcessor(subarray, use_telescope_frame=False)
    reconstructor = HillasReconstructor(subarray)

    # Get shower geometry
    calib(event)
    image_processor(event)
    result = reconstructor(event)

    # get the reconstructed coordinates in the sky
    reco_coord = SkyCoord(alt=result.alt, az=result.az, frame=AltAz())
    # get the simulated coordinates in the sky
    true_coord = SkyCoord(
        alt=event.simulation.shower.alt, az=event.simulation.shower.az, frame=AltAz()
    )

    # check that we are not more far than 0.1 degrees
    assert reco_coord.separation(true_coord) < 0.1 * u.deg

    assert u.isclose(result.core_x, event.simulation.shower.core_x, atol=25 * u.m)
    assert u.isclose(result.core_y, event.simulation.shower.core_y, atol=25 * u.m)


@pytest.mark.parametrize(
    "filename",
    [
        "gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz",
        "gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz",
    ],
)
def test_CameraFrame_against_TelescopeFrame(filename):

    input_file = get_dataset_path(filename)
    # "gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz"
    # )

    source = SimTelEventSource(
        input_file, max_events=10, focal_length_choice="EQUIVALENT"
    )

    # too few events survive for this test with the defautl quality criteria,
    # use less restrictive ones
    config = Config(
        {
            "StereoQualityQuery": {
                "quality_criteria": [
                    ("valid_width", "parameters.hillas.width.value > 0"),
                ]
            }
        }
    )

    calib = CameraCalibrator(subarray=source.subarray)
    reconstructor = HillasReconstructor(source.subarray, config=config)
    image_processor_camera_frame = ImageProcessor(
        source.subarray, use_telescope_frame=False
    )
    image_processor_telescope_frame = ImageProcessor(
        source.subarray, use_telescope_frame=True
    )

    reconstructed_events = 0

    for event_telescope_frame in source:

        calib(event_telescope_frame)
        # make a copy of the calibrated event for the camera frame case
        # later we clean and paramretrize the 2 events in the same way
        # but in 2 different frames to check they return compatible results
        event_camera_frame = deepcopy(event_telescope_frame)

        image_processor_telescope_frame(event_telescope_frame)
        image_processor_camera_frame(event_camera_frame)

        reconstructor(event_camera_frame)
        result_camera_frame = event_camera_frame.dl2.stereo.geometry[
            "HillasReconstructor"
        ]
        reconstructor(event_telescope_frame)
        result_telescope_frame = event_telescope_frame.dl2.stereo.geometry[
            "HillasReconstructor"
        ]

        assert result_camera_frame.is_valid == result_telescope_frame.is_valid

        if result_telescope_frame.is_valid:
            reconstructed_events += 1

            for field, cam in result_camera_frame.items():
                tel = getattr(result_telescope_frame, field)

                if hasattr(cam, "unit"):
                    assert u.isclose(
                        cam, tel, rtol=1e-3, atol=1e-3 * tel.unit, equal_nan=True
                    )
                elif isinstance(cam, list):
                    assert cam == tel
                else:
                    assert np.isclose(cam, tel, rtol=1e-3, atol=1e-3, equal_nan=True)

    assert reconstructed_events > 0  # check that we reconstruct at least 1 event
