import numpy as np
from astropy import units as u
import pytest

from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.io import event_source
from ctapipe.reco.HillasReconstructor import HillasReconstructor, HillasPlane
from ctapipe.reco.reco_algorithms import (
    TooFewTelescopesException,
    InvalidWidthException,
)
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

    # creating the fit class and setting the the great circle member
    fit = HillasReconstructor()
    fit.hillas_planes = {1: circle1, 2: circle2, 3: circle3, 4: circle4}

    # performing the direction fit with the minimisation algorithm
    # and a seed that is perpendicular to the up direction
    dir_fit_minimise, _ = fit.estimate_direction()
    print("direction fit test minimise:", dir_fit_minimise)
    print()


def test_h_max_results():
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

    # creating the fit class and setting the the great circle member
    fit = HillasReconstructor()
    fit.hillas_planes = {1: circle1, 2: circle2, 3: circle3, 4: circle4}

    # performing the direction fit with the minimisation algorithm
    # and a seed that is perpendicular to the up direction
    h_max_reco = fit.estimate_h_max()
    print("h max fit test minimise:", h_max_reco)

    # the results should be close to the direction straight up
    np.testing.assert_allclose(h_max_reco.value, 0, atol=1e-8)
    # np.testing.assert_allclose(fitted_core_position.value, [0, 0], atol=1e-3)


def test_parallel_reconstruction():
    """
    Test shower's reconstruction procedure:
    • image cleaning
    • hillas parametrisation
    • HillasPlane creation
    • shower direction reconstruction in the sky
    • shower core reconstruction in the ground

    Tested,
    - starting from CameraFrame,
    - starting from TelescopeFrame,

    The test checks that the old approach (using CameraFrame) and the new one
    (using TelescopeFrame) provide compatible results and that we are able to
    reconstruct a positive number of events.
    """
    from scipy.spatial import distance

    filename = get_dataset_path(
        "gamma_LaPalma_baseline_20Zd_180Az_prod3b_test.simtel.gz"
    )

    source = event_source(filename, max_events=10)
    calib = CameraCalibrator(subarray=source.subarray)

    horizon_frame = AltAz()

    reconstructed_events = 0

    # ==========================================================================

    for event in source:
        calib(event)

        mc = event.simulation.shower
        array_pointing = SkyCoord(az=mc.az, alt=mc.alt, frame=horizon_frame)

        hillas_dict_CameraFrame = {}
        hillas_dict_TelescopeFrame = {}
        telescope_pointings = {}

        for tel_id, dl1 in event.dl1.tel.items():

            telescope_pointings[tel_id] = SkyCoord(
                alt=event.pointing.tel[tel_id].altitude,
                az=event.pointing.tel[tel_id].azimuth,
                frame=horizon_frame,
            )

            geom_CameraFrame = source.subarray.tel[tel_id].camera.geometry

            # this could be done also out of this loop,
            # but in case of real data each telescope would have a
            # different telescope_pointing
            geom_TelescopeFrame = geom_CameraFrame.transform_to(
                TelescopeFrame(telescope_pointing=telescope_pointings[tel_id])
            )

            mask = tailcuts_clean(
                geom_TelescopeFrame,
                dl1.image,
                picture_thresh=10.0,
                boundary_thresh=5.0,
            )

            try:
                moments_CameraFrame = hillas_parameters(geom_CameraFrame[mask], dl1.image[mask])
                moments_TelescopeFrame = hillas_parameters(
                    geom_TelescopeFrame[mask], dl1.image[mask]
                )
                hillas_dict_CameraFrame[tel_id] = moments_CameraFrame
                hillas_dict_TelescopeFrame[tel_id] = moments_TelescopeFrame
            except HillasParameterizationError as e:
                print(e)
                continue

        if (len(hillas_dict_CameraFrame) < 2) and (len(hillas_dict_TelescopeFrame) < 2):
            continue
        else:
            reconstructed_events += 1

        try:

            # Parallel pointing case using CameraFrame
            fit = HillasReconstructor(event=event)
            fit_result_CameraFrame = fit.predict(
                hillas_dict_CameraFrame, source.subarray, array_pointing, telescope_pointings
            )

            # Parallel pointing case using TelescopeFrame
            fit = HillasReconstructor(event=event)
            fit_result_TelescopeFrame = fit.predict(
                hillas_dict_TelescopeFrame, source.subarray, array_pointing, telescope_pointings
            )

        except InvalidWidthException:
            continue

        for field in fit_result_CameraFrame.as_dict():
            C = np.asarray(fit_result_CameraFrame.as_dict()[field])
            T = np.asarray(fit_result_TelescopeFrame.as_dict()[field])
            assert (np.isclose(C, T, rtol=1e-03, atol=1e-03, equal_nan=True)).all()

    assert reconstructed_events > 0


def test_divergent_reconstruction():
    """
    Test shower's reconstruction procedure:
    • image cleaning
    • hillas parametrisation
    • HillasPlane creation
    • shower direction reconstruction in the sky
    • shower core reconstruction in the ground

    Tested,
    - starting from CameraFrame,
    - starting from TelescopeFrame,
    - divergent pointing using divergent pointing test data

    The test checks the old approach (using CameraFrame) and the new one
    (using TelescopeFrame) provide compatible results and that we are able to
    reconstruct a positive number of events.
    """
    from scipy.spatial import distance

    filename = get_dataset_path(
        "gamma_divergent_LaPalma_baseline_20Zd_180Az_prod3_test.simtel.gz"
    )

    source = event_source(filename, max_events=10)
    horizon_frame = AltAz()

    reconstructed_events = 0

    # ==========================================================================

    for event in source:

        array_pointing = SkyCoord(
            az=event.pointing.array_azimuth,
            alt=event.pointing.array_altitude,
            frame=horizon_frame,
        )

        hillas_dict_CameraFrame = {}
        hillas_dict_TelescopeFrame = {}
        telescope_pointings = {}

        for tel_id in event.r1.tel.keys():

            telescope_pointings[tel_id] = SkyCoord(
                alt=event.pointing.tel[tel_id].altitude,
                az=event.pointing.tel[tel_id].azimuth,
                frame=horizon_frame,
            )

            geom_CameraFrame = source.subarray.tel[tel_id].camera.geometry

            # this could be done also out of this loop,
            # but in case of real data each telescope would have a
            # different telescope_pointing
            geom_TelescopeFrame = geom_CameraFrame.transform_to(
                TelescopeFrame(telescope_pointing=telescope_pointings[tel_id])
            )

            pmt_signal = event.r0.tel[tel_id].waveform[0].sum(axis=1)

            mask = tailcuts_clean(
                geom_TelescopeFrame,
                pmt_signal,
                picture_thresh=10.0,
                boundary_thresh=5.0,
            )
            pmt_signal[mask == 0] = 0

            try:
                moments_CameraFrame = hillas_parameters(geom_CameraFrame, pmt_signal)
                moments_TelescopeFrame = hillas_parameters(
                    geom_TelescopeFrame, pmt_signal
                )
                hillas_dict_CameraFrame[tel_id] = moments_CameraFrame
                hillas_dict_TelescopeFrame[tel_id] = moments_TelescopeFrame
            except HillasParameterizationError as e:
                print(e)
                continue

        if (len(hillas_dict_CameraFrame) < 2) and (len(hillas_dict_TelescopeFrame) < 2):
            continue
        else:
            reconstructed_events += 1

        fit = HillasReconstructor(event=event)
        fit_result_CameraFrame = fit.predict(
            hillas_dict_CameraFrame,
            source.subarray,
            array_pointing,
            telescope_pointings,
        )

        fit = HillasReconstructor(event=event)
        fit_result_TelescopeFrame = fit.predict(
            hillas_dict_TelescopeFrame,
            source.subarray,
            array_pointing,
            telescope_pointings,
        )

        # Compare old approach with new approach
        for field in fit_result_CameraFrame.as_dict():
            C = np.asarray(fit_result_CameraFrame.as_dict()[field])
            T = np.asarray(fit_result_TelescopeFrame.as_dict()[field])
            assert (np.isclose(C, T, rtol=1e-03, atol=1e-03, equal_nan=True)).all()

    assert reconstructed_events > 0


def test_invalid_events():
    """
    The HillasReconstructor is supposed to fail
    in these cases:
    - less than two teleskopes
    - any width is NaN
    - any width is 0

    This test uses the same sample simtel file as
    test_reconstruction(). As there are no invalid events in this
    file, multiple hillas_dicts are constructed to make sure
    Exceptions get thrown in the mentioned edge cases.

    Test will fail if no Exception or another Exception gets thrown."""

    filename = get_dataset_path("gamma_test_large.simtel.gz")

    tel_azimuth = {}
    tel_altitude = {}

    source = event_source(filename, max_events=10)
    subarray = source.subarray
    calib = CameraCalibrator(subarray)

    for event in source:
        fit = HillasReconstructor(event=event)
        calib(event)

        hillas_dict = {}
        for tel_id, dl1 in event.dl1.tel.items():

            geom = source.subarray.tel[tel_id].camera.geometry
            tel_azimuth[tel_id] = event.pointing.tel[tel_id].azimuth
            tel_altitude[tel_id] = event.pointing.tel[tel_id].altitude

            mask = tailcuts_clean(
                geom, dl1.image, picture_thresh=10.0, boundary_thresh=5.0
            )

            try:
                moments = hillas_parameters(geom[mask], dl1.image[mask])
                hillas_dict[tel_id] = moments
            except HillasParameterizationError as e:
                continue

        # construct a dict only containing the last telescope events
        # (#telescopes < 2)
        hillas_dict_only_one_tel = dict()
        hillas_dict_only_one_tel[tel_id] = hillas_dict[tel_id]
        with pytest.raises(TooFewTelescopesException):
            fit.predict(hillas_dict_only_one_tel, subarray, tel_azimuth, tel_altitude)

        # construct a hillas dict with the width of the last event set to 0
        # (any width == 0)
        hillas_dict_zero_width = hillas_dict.copy()
        hillas_dict_zero_width[tel_id]["width"] = 0 * u.m
        with pytest.raises(InvalidWidthException):
            fit.predict(hillas_dict_zero_width, subarray, tel_azimuth, tel_altitude)

        # construct a hillas dict with the width of the last event set to np.nan
        # (any width == nan)
        hillas_dict_nan_width = hillas_dict.copy()
        hillas_dict_zero_width[tel_id]["width"] = np.nan * u.m
        with pytest.raises(InvalidWidthException):
            fit.predict(hillas_dict_nan_width, subarray, tel_azimuth, tel_altitude)
