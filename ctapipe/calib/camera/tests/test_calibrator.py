"""
Tests for CameraCalibrator and related functions
"""
import astropy.units as u
import numpy as np
import pytest
from scipy.stats import norm
from traitlets.config.configurable import Config

from ctapipe.calib.camera.calibrator import CameraCalibrator
from ctapipe.image.extractor import (
    NeighborPeakWindowSum,
    LocalPeakWindowSum,
    FullWaveformSum,
    GlobalPeakWindowSum,
)
from ctapipe.image.reducer import NullDataVolumeReducer, TailCutsDataVolumeReducer
from copy import deepcopy
from ctapipe.containers import ArrayEventContainer


def test_camera_calibrator(example_event, example_subarray):
    telid = list(example_event.r0.tel)[0]
    calibrator = CameraCalibrator(subarray=example_subarray)
    calibrator(example_event)
    image = example_event.dl1.tel[telid].image
    peak_time = example_event.dl1.tel[telid].peak_time
    assert image is not None
    assert peak_time is not None
    assert image.shape == (1764,)
    assert peak_time.shape == (1764,)


def test_manual_extractor(example_subarray):
    calibrator = CameraCalibrator(
        subarray=example_subarray,
        image_extractor=LocalPeakWindowSum(subarray=example_subarray),
    )
    assert isinstance(calibrator.image_extractor, LocalPeakWindowSum)


def test_config(example_subarray):
    calibrator = CameraCalibrator(subarray=example_subarray)

    # test defaults
    assert isinstance(calibrator.image_extractor, NeighborPeakWindowSum)
    assert isinstance(calibrator.data_volume_reducer, NullDataVolumeReducer)

    config = Config(
        {
            "CameraCalibrator": {
                "image_extractor_type": "LocalPeakWindowSum",
                "LocalPeakWindowSum": {"window_width": 15},
                "data_volume_reducer_type": "TailCutsDataVolumeReducer",
                "TailCutsDataVolumeReducer": {
                    "TailcutsImageCleaner": {"picture_threshold_pe": 20.0}
                },
            }
        }
    )

    calibrator = CameraCalibrator(example_subarray, config=config)
    assert isinstance(calibrator.image_extractor, LocalPeakWindowSum)
    assert calibrator.image_extractor.window_width.tel[None] == 15

    assert isinstance(calibrator.data_volume_reducer, TailCutsDataVolumeReducer)
    assert calibrator.data_volume_reducer.cleaner.picture_threshold_pe.tel[None] == 20


def test_check_r1_empty(example_event, example_subarray):
    calibrator = CameraCalibrator(subarray=example_subarray)
    telid = list(example_event.r0.tel)[0]
    waveform = example_event.r1.tel[telid].waveform.copy()
    with pytest.warns(UserWarning):
        example_event.r1.tel[telid].waveform = None
        calibrator._calibrate_dl0(example_event, telid)
        assert example_event.dl0.tel[telid].waveform is None

    assert calibrator._check_r1_empty(None) is True
    assert calibrator._check_r1_empty(waveform) is False

    calibrator = CameraCalibrator(
        subarray=example_subarray,
        image_extractor=FullWaveformSum(subarray=example_subarray),
    )
    event = ArrayEventContainer()
    event.dl0.tel[telid].waveform = np.full((2048, 128), 2)
    with pytest.warns(UserWarning):
        calibrator(event)
    assert (event.dl0.tel[telid].waveform == 2).all()
    assert (event.dl1.tel[telid].image == 2 * 128).all()


def test_check_dl0_empty(example_event, example_subarray):
    calibrator = CameraCalibrator(subarray=example_subarray)
    telid = list(example_event.r0.tel)[0]
    calibrator._calibrate_dl0(example_event, telid)
    waveform = example_event.dl0.tel[telid].waveform.copy()
    with pytest.warns(UserWarning):
        example_event.dl0.tel[telid].waveform = None
        calibrator._calibrate_dl1(example_event, telid)
        assert example_event.dl1.tel[telid].image is None

    assert calibrator._check_dl0_empty(None) is True
    assert calibrator._check_dl0_empty(waveform) is False

    calibrator = CameraCalibrator(subarray=example_subarray)
    event = ArrayEventContainer()
    event.dl1.tel[telid].image = np.full(2048, 2)
    with pytest.warns(UserWarning):
        calibrator(event)
    assert (event.dl1.tel[telid].image == 2).all()


def test_dl1_charge_calib(example_subarray):
    # copy because we mutate the camera, should not affect other tests
    subarray = deepcopy(example_subarray)
    camera = subarray.tel[1].camera
    # test with a sampling_rate different than 1 to
    # test if we handle time vs. slices correctly
    sampling_rate = 2
    camera.readout.sampling_rate = sampling_rate * u.GHz

    n_pixels = camera.geometry.n_pixels
    n_samples = 96
    mid = n_samples // 2
    pulse_sigma = 6
    random = np.random.RandomState(1)
    x = np.arange(n_samples)

    # Randomize times and create pulses
    time_offset = random.uniform(-10, +10, n_pixels)
    y = norm.pdf(x, mid + time_offset[:, np.newaxis], pulse_sigma).astype("float32")

    camera.readout.reference_pulse_shape = norm.pdf(x, mid, pulse_sigma)[np.newaxis, :]
    camera.readout.reference_pulse_sample_width = 1 / camera.readout.sampling_rate

    # Define absolute calibration coefficients
    absolute = random.uniform(100, 1000, n_pixels).astype("float32")
    y *= absolute[:, np.newaxis]

    # Define relative coefficients
    relative = random.normal(1, 0.01, n_pixels)
    y /= relative[:, np.newaxis]

    # Define pedestal
    pedestal = random.uniform(-4, 4, n_pixels)
    y += pedestal[:, np.newaxis]

    event = ArrayEventContainer()
    telid = list(subarray.tel.keys())[0]
    event.dl0.tel[telid].waveform = y
    event.dl0.tel[telid].selected_gain_channel = np.zeros(len(y), dtype=int)
    event.r1.tel[telid].selected_gain_channel = np.zeros(len(y), dtype=int)

    # Test default
    calibrator = CameraCalibrator(
        subarray=subarray, image_extractor=FullWaveformSum(subarray=subarray)
    )
    calibrator(event)
    np.testing.assert_allclose(event.dl1.tel[telid].image, y.sum(1), rtol=1e-4)

    event.calibration.tel[telid].dl1.pedestal_offset = pedestal
    event.calibration.tel[telid].dl1.absolute_factor = absolute
    event.calibration.tel[telid].dl1.relative_factor = relative

    # Test without timing corrections
    calibrator(event)
    dl1 = event.dl1.tel[telid]
    np.testing.assert_allclose(dl1.image, 1, rtol=1e-5)
    expected_peak_time = (mid + time_offset) / sampling_rate
    np.testing.assert_allclose(dl1.peak_time, expected_peak_time, rtol=1e-5)

    # test with timing corrections
    event.calibration.tel[telid].dl1.time_shift = time_offset / sampling_rate
    calibrator(event)

    # more rtol since shifting might lead to reduced integral
    np.testing.assert_allclose(event.dl1.tel[telid].image, 1, rtol=1e-5)
    np.testing.assert_allclose(
        event.dl1.tel[telid].peak_time, mid / sampling_rate, atol=1
    )

    # test not applying time shifts
    # now we should be back to the result without setting time shift
    calibrator.apply_peak_time_shift = False
    calibrator.apply_waveform_time_shift = False
    calibrator(event)

    np.testing.assert_allclose(event.dl1.tel[telid].image, 1, rtol=1e-4)
    np.testing.assert_allclose(
        event.dl1.tel[telid].peak_time, expected_peak_time, atol=1
    )

    # We now use GlobalPeakWindowSum to see the effect of missing charge
    # due to not correcting time offsets.
    calibrator = CameraCalibrator(
        subarray=subarray, image_extractor=GlobalPeakWindowSum(subarray=subarray)
    )
    calibrator(event)
    # test with timing corrections, should work
    # higher rtol because we cannot shift perfectly
    np.testing.assert_allclose(event.dl1.tel[telid].image, 1, rtol=0.01)
    np.testing.assert_allclose(
        event.dl1.tel[telid].peak_time, mid / sampling_rate, atol=1
    )

    # test deactivating timing corrections
    calibrator.apply_waveform_time_shift = False
    calibrator(event)

    # make sure we chose an example where the time shifts matter
    # charges should be quite off due to summing around global shift
    assert not np.allclose(event.dl1.tel[telid].image, 1, rtol=0.1)
    assert not np.allclose(event.dl1.tel[telid].peak_time, mid / sampling_rate, atol=1)


def test_shift_waveforms():
    from ctapipe.calib.camera.calibrator import shift_waveforms

    # 5 pixels, 40 samples
    waveforms = np.zeros((5, 40))
    waveforms[:, 10] = 1
    shifts = np.array([1.4, 2.1, -1.8, 3.1, -4.4])

    shifted_waveforms, remaining_shift = shift_waveforms(waveforms, shifts)

    assert np.allclose(remaining_shift, [0.4, 0.1, 0.2, 0.1, -0.4])

    assert shifted_waveforms[0, 9] == 1
    assert shifted_waveforms[1, 8] == 1
    assert shifted_waveforms[2, 12] == 1
    assert shifted_waveforms[3, 7] == 1
    assert shifted_waveforms[4, 14] == 1
