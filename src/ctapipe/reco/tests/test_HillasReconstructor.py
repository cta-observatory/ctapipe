from copy import deepcopy

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from traitlets.config import Config

from ctapipe.calib import CameraCalibrator
from ctapipe.containers import HillasParametersContainer
from ctapipe.coordinates import GroundFrame, altaz_to_righthanded_cartesian
from ctapipe.image.image_processor import ImageProcessor
from ctapipe.io import SimTelEventSource
from ctapipe.reco.hillas_reconstructor import HillasReconstructor
from ctapipe.utils import get_dataset_path


def test_estimator_results():
    """Test direction results from cog and point 2 (along the main axis) coordinates"""

    weight = np.ones(4)
    cog_alt = [43, 44, 44.5, 43.5] * u.deg
    cog_az = [45, 90, 45, 90] * u.deg

    p2_alt = [47, 45, 46.5, 45.5] * u.deg
    p2_az = [45, 90, 45, 90] * u.deg

    cog_cart = altaz_to_righthanded_cartesian(cog_alt, cog_az)
    p2_cart = altaz_to_righthanded_cartesian(p2_alt, p2_az)
    norm = np.cross(cog_cart, p2_cart)

    direction, _ = HillasReconstructor.estimate_direction(norm, weight)
    assert np.allclose(direction, [0, 0, 1])


def test_h_max_results():
    """test h_max estimation from cog coordinates"""
    cog_alt = [45.0, 45.0, 45.0, 45.0] * u.deg
    cog_az = [180, 90, 0, -90.0] * u.deg

    positions = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]] * u.km
    positions = SkyCoord(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], frame=GroundFrame()
    )

    cog_cart = altaz_to_righthanded_cartesian(cog_alt, cog_az)
    h_max_reco = HillasReconstructor.estimate_relative_h_max(
        cog_vectors=cog_cart, positions=positions
    )

    # the results should be close to the direction straight up
    assert u.isclose(h_max_reco, 1 * u.km)


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
    # test the container is actually there and not only created by Map
    assert "HillasReconstructor" in event.dl2.stereo.geometry
    result = event.dl2.stereo.geometry["HillasReconstructor"]
    assert result.is_valid

    # copy event container to modify it
    invalid_event = deepcopy(original_event)

    # overwrite all image parameters but the last one with dummy ones
    for tel_id in list(invalid_event.dl1.tel.keys())[:-1]:
        invalid_event.dl1.tel[tel_id].parameters.hillas = HillasParametersContainer()

    hillas_reconstructor(invalid_event)
    # test the container is actually there and not only created by Map
    assert "HillasReconstructor" in invalid_event.dl2.stereo.geometry
    result = invalid_event.dl2.stereo.geometry["HillasReconstructor"]
    assert not result.is_valid

    tel_id = list(invalid_event.dl1.tel.keys())[-1]
    # Now use the original event, but overwrite the last width to 0
    invalid_event = deepcopy(original_event)
    invalid_event.dl1.tel[tel_id].parameters.hillas.width = 0 * u.m
    hillas_reconstructor(invalid_event)
    # test the container is actually there and not only created by Map
    assert "HillasReconstructor" in invalid_event.dl2.stereo.geometry
    result = invalid_event.dl2.stereo.geometry["HillasReconstructor"]
    assert not result.is_valid

    # Now use the original event, but overwrite the last width to NaN
    invalid_event = deepcopy(original_event)
    invalid_event.dl1.tel[tel_id].parameters.hillas.width = np.nan * u.m
    hillas_reconstructor(invalid_event)
    # test the container is actually there and not only created by Map
    assert "HillasReconstructor" in invalid_event.dl2.stereo.geometry
    result = invalid_event.dl2.stereo.geometry["HillasReconstructor"]
    assert not result.is_valid


def test_reconstruction_against_simulation_telescope_frame(
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
    image_processor = ImageProcessor(subarray)
    reconstructor = HillasReconstructor(subarray)

    # Get shower geometry
    calib(event)
    image_processor(event)
    reconstructor(event)
    # test the container is actually there and not only created by Map
    assert "HillasReconstructor" in event.dl2.stereo.geometry
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
    """Reconstruction is here done only in the CameraFrame,
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
    reconstructor(event)
    result = event.dl2.stereo.geometry[reconstructor.__class__.__name__]

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

    # too few events survive for this test with the default quality criteria,
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

                kwargs = dict(rtol=6e-3, equal_nan=True)

                if hasattr(cam, "unit"):
                    if cam.value == 0 or tel.value == 0:
                        kwargs["atol"] = 1e-6 * cam.unit
                    assert u.isclose(
                        cam, tel, **kwargs
                    ), f"attr {field} not matching, camera: {result_camera_frame!s} telescope: {result_telescope_frame!s}"
                elif isinstance(cam, list):
                    assert cam == tel
                else:
                    if cam == 0 or tel == 0:
                        kwargs["atol"] = 1e-6
                    assert np.isclose(cam, tel, **kwargs)

    assert reconstructed_events > 0  # check that we reconstruct at least 1 event


def test_angle():
    from ctapipe.reco.hillas_reconstructor import angle

    # test it works with single vectors
    assert np.isclose(angle(np.array([0, 0, 1]), np.array([1, 0, 0])), np.pi / 2)

    # and with an array of vectors
    a = np.array([[1, 0, 0], [1, 0, 0]])
    b = np.array([[1, 0, 0], [0, 1, 0]])
    assert np.allclose(angle(a, b), [0, np.pi / 2])
