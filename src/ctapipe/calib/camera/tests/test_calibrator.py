"""
Tests for CameraCalibrator and related functions
"""

from copy import deepcopy

import astropy.units as u
import numpy as np
from scipy.stats import norm
from traitlets.config import Config

from ctapipe.calib.camera.calibrator import CameraCalibrator
from ctapipe.containers import ArrayEventContainer
from ctapipe.image.extractor import (
    FullWaveformSum,
    GlobalPeakWindowSum,
    LocalPeakWindowSum,
    NeighborPeakWindowSum,
    VarianceExtractor,
)
from ctapipe.image.reducer import NullDataVolumeReducer, TailCutsDataVolumeReducer


def test_camera_calibrator(example_event, example_subarray):
    tel_id = list(example_event.r0.tel)[0]
    calibrator = CameraCalibrator(subarray=example_subarray)
    calibrator(example_event)
    image = example_event.dl1.tel[tel_id].image
    peak_time = example_event.dl1.tel[tel_id].peak_time
    assert image is not None
    assert peak_time is not None
    assert image.shape == (1764,)
    assert peak_time.shape == (1764,)


def test_manual_extractor(example_subarray):
    extractor = LocalPeakWindowSum(subarray=example_subarray)
    calibrator = CameraCalibrator(subarray=example_subarray, image_extractor=extractor)
    assert "LocalPeakWindowSum" in calibrator.image_extractors
    assert calibrator.image_extractor_type.tel[1] == "LocalPeakWindowSum"
    assert calibrator.image_extractors["LocalPeakWindowSum"] is extractor


def test_config(example_subarray):
    calibrator = CameraCalibrator(subarray=example_subarray)

    # test defaults
    assert len(calibrator.image_extractors) == 1
    assert isinstance(
        calibrator.image_extractors["NeighborPeakWindowSum"], NeighborPeakWindowSum
    )
    assert isinstance(calibrator.data_volume_reducer, NullDataVolumeReducer)

    # test we can configure different extractors with different options
    # per telescope.
    config = Config(
        {
            "CameraCalibrator": {
                "image_extractor_type": [
                    ("type", "*", "GlobalPeakWindowSum"),
                    ("id", 1, "LocalPeakWindowSum"),
                ],
                "LocalPeakWindowSum": {"window_width": 15},
                "GlobalPeakWindowSum": {
                    "window_width": [("type", "*", 10), ("id", 2, 8)]
                },
                "data_volume_reducer_type": "TailCutsDataVolumeReducer",
                "TailCutsDataVolumeReducer": {
                    "TailcutsImageCleaner": {"picture_threshold_pe": 20.0}
                },
            }
        }
    )

    calibrator = CameraCalibrator(example_subarray, config=config)
    assert "GlobalPeakWindowSum" in calibrator.image_extractors
    assert "LocalPeakWindowSum" in calibrator.image_extractors
    assert isinstance(
        calibrator.image_extractors["LocalPeakWindowSum"], LocalPeakWindowSum
    )
    assert isinstance(
        calibrator.image_extractors["GlobalPeakWindowSum"], GlobalPeakWindowSum
    )

    extractor_1 = calibrator.image_extractors[calibrator.image_extractor_type.tel[1]]
    assert isinstance(extractor_1, LocalPeakWindowSum)
    assert extractor_1.window_width.tel[1] == 15

    extractor_2 = calibrator.image_extractors[calibrator.image_extractor_type.tel[2]]
    assert isinstance(extractor_2, GlobalPeakWindowSum)
    assert extractor_2.window_width.tel[2] == 8

    extractor_3 = calibrator.image_extractors[calibrator.image_extractor_type.tel[3]]
    assert isinstance(extractor_3, GlobalPeakWindowSum)
    assert extractor_3.window_width.tel[3] == 10

    assert isinstance(calibrator.data_volume_reducer, TailCutsDataVolumeReducer)
    assert calibrator.data_volume_reducer.cleaner.picture_threshold_pe.tel[None] == 20


def test_check_r1_empty(example_event, example_subarray):
    calibrator = CameraCalibrator(subarray=example_subarray)
    tel_id = list(example_event.r0.tel)[0]
    waveform = example_event.r1.tel[tel_id].waveform.copy()
    assert calibrator._check_r1_empty(None) is True
    assert calibrator._check_r1_empty(waveform) is False

    calibrator = CameraCalibrator(
        subarray=example_subarray,
        image_extractor=FullWaveformSum(subarray=example_subarray),
    )
    event = ArrayEventContainer()
    event.dl0.tel[tel_id].waveform = np.full((1, 2048, 128), 2)
    calibrator(event)
    assert (event.dl0.tel[tel_id].waveform == 2).all()
    assert (event.dl1.tel[tel_id].image == 2 * 128).all()


def test_check_dl0_empty(example_event, example_subarray):
    calibrator = CameraCalibrator(subarray=example_subarray)
    tel_id = list(example_event.r0.tel)[0]
    calibrator._calibrate_dl0(example_event, tel_id)
    waveform = example_event.dl0.tel[tel_id].waveform.copy()
    assert calibrator._check_dl0_empty(None) is True
    assert calibrator._check_dl0_empty(waveform) is False

    calibrator = CameraCalibrator(subarray=example_subarray)
    event = ArrayEventContainer()
    event.dl1.tel[tel_id].image = np.full(2048, 2)
    calibrator(event)
    assert (event.dl1.tel[tel_id].image == 2).all()


def test_dl1_variance_calib(example_subarray):
    calibrator = CameraCalibrator(
        subarray=example_subarray,
        image_extractor=VarianceExtractor(subarray=example_subarray),
        apply_waveform_time_shift=False,
    )
    n_samples = 100

    event = ArrayEventContainer()

    for tel_type in example_subarray.telescope_types:
        tel_id = example_subarray.get_tel_ids_for_type(tel_type)[0]
        n_pixels = example_subarray.tel[tel_id].camera.geometry.n_pixels
        n_channels = example_subarray.tel[tel_id].camera.readout.n_channels

        random = np.random.default_rng(1)
        y = random.normal(0, 6, (n_channels, n_pixels, n_samples))

        absolute = random.uniform(100, 1000, (n_channels, n_pixels)).astype("float32")
        y *= absolute[..., np.newaxis]

        relative = random.normal(1, 0.01, (n_channels, n_pixels))
        y /= relative[..., np.newaxis]

        pedestal = random.uniform(-4, 4, (n_channels, n_pixels))
        y += pedestal[..., np.newaxis]

        event.dl0.tel[tel_id].waveform = y
        event.calibration.tel[tel_id].dl1.pedestal_offset = pedestal
        event.calibration.tel[tel_id].dl1.absolute_factor = absolute
        event.calibration.tel[tel_id].dl1.relative_factor = relative
        event.dl0.tel[tel_id].selected_gain_channel = None
        event.r1.tel[tel_id].selected_gain_channel = None

    calibrator(event)

    for tel_type in example_subarray.telescope_types:
        tel_id = example_subarray.get_tel_ids_for_type(tel_type)[0]
        image = event.dl1.tel[tel_id].image
        camera = example_subarray.tel[tel_id].camera
        assert image is not None
        assert image.shape == (
            camera.readout.n_channels,
            example_subarray.tel[tel_id].camera.geometry.n_pixels,
        )


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
    random = np.random.default_rng(1)
    x = np.arange(n_samples)

    camera.readout.reference_pulse_shape = norm.pdf(x, mid, pulse_sigma)
    camera.readout.reference_pulse_shape = np.repeat(
        camera.readout.reference_pulse_shape[np.newaxis, :], 2, axis=0
    )
    camera.readout.reference_pulse_sample_width = 1 / camera.readout.sampling_rate

    # test that it works for 1 and 2 gain channel
    gain_channel = [1, 2]
    for n_channels in gain_channel:
        # Randomize times and create pulses
        time_offset = random.uniform(-10, +10, (n_channels, n_pixels))
        y = norm.pdf(x, mid + time_offset[..., np.newaxis], pulse_sigma).astype(
            "float32"
        )

        # Define absolute calibration coefficients
        absolute = random.uniform(100, 1000, (n_channels, n_pixels)).astype("float32")
        y *= absolute[..., np.newaxis]

        # Define relative coefficients
        relative = random.normal(1, 0.01, (n_channels, n_pixels))
        y /= relative[..., np.newaxis]

        # Define pedestal
        pedestal = random.uniform(-4, 4, (n_channels, n_pixels))
        y += pedestal[..., np.newaxis]

        event = ArrayEventContainer()
        tel_id = list(subarray.tel.keys())[0]
        event.dl0.tel[tel_id].waveform = y
        selected_gain_channel = None
        if n_channels == 1:
            selected_gain_channel = np.zeros(n_pixels, dtype=int)
        event.dl0.tel[tel_id].selected_gain_channel = selected_gain_channel
        event.r1.tel[tel_id].selected_gain_channel = selected_gain_channel

        # Test default
        calibrator = CameraCalibrator(
            subarray=subarray, image_extractor=FullWaveformSum(subarray=subarray)
        )
        calibrator(event)
        np.testing.assert_allclose(
            event.dl1.tel[tel_id].image, y.sum(-1).squeeze(), rtol=1e-4
        )

        event.calibration.tel[tel_id].dl1.pedestal_offset = pedestal
        event.calibration.tel[tel_id].dl1.absolute_factor = absolute
        event.calibration.tel[tel_id].dl1.relative_factor = relative

        # Test without timing corrections
        calibrator(event)
        dl1 = event.dl1.tel[tel_id]
        np.testing.assert_allclose(dl1.image, 1, rtol=1e-5)

        expected_peak_time = (mid + time_offset) / sampling_rate
        np.testing.assert_allclose(
            dl1.peak_time, expected_peak_time.squeeze(), rtol=1e-5
        )

        # test with timing corrections
        event.calibration.tel[tel_id].dl1.time_shift = time_offset / sampling_rate
        calibrator(event)

        # more rtol since shifting might lead to reduced integral
        np.testing.assert_allclose(event.dl1.tel[tel_id].image, 1, rtol=1e-5)
        np.testing.assert_allclose(
            event.dl1.tel[tel_id].peak_time, mid / sampling_rate, atol=1
        )

        # test not applying time shifts
        # now we should be back to the result without setting time shift
        calibrator.apply_peak_time_shift = False
        calibrator.apply_waveform_time_shift = False
        calibrator(event)

        np.testing.assert_allclose(event.dl1.tel[tel_id].image, 1, rtol=1e-4)
        np.testing.assert_allclose(
            event.dl1.tel[tel_id].peak_time, expected_peak_time.squeeze(), atol=1
        )

        # We now use GlobalPeakWindowSum to see the effect of missing charge
        # due to not correcting time offsets.
        calibrator = CameraCalibrator(
            subarray=subarray,
            image_extractor=GlobalPeakWindowSum(subarray=subarray),
            apply_waveform_time_shift=True,
        )
        calibrator(event)
        # test with timing corrections, should work
        # higher rtol because we cannot shift perfectly
        np.testing.assert_allclose(event.dl1.tel[tel_id].image, 1, rtol=0.01)
        np.testing.assert_allclose(
            event.dl1.tel[tel_id].peak_time, mid / sampling_rate, atol=1
        )

        # test deactivating timing corrections
        calibrator.apply_waveform_time_shift = False
        calibrator(event)

        # make sure we chose an example where the time shifts matter
        # charges should be quite off due to summing around global shift
        assert not np.allclose(event.dl1.tel[tel_id].image, 1, rtol=0.1)
        assert not np.allclose(
            event.dl1.tel[tel_id].peak_time, mid / sampling_rate, atol=1
        )


def test_shift_waveforms():
    from ctapipe.calib.camera.calibrator import shift_waveforms

    # 1 channel, 5 pixels, 40 samples
    waveforms = np.zeros((1, 5, 40))
    waveforms[:, :, 10] = 1
    shifts = np.array([1.4, 2.1, -1.8, 3.1, -4.4])

    shifted_waveforms, remaining_shift = shift_waveforms(waveforms, shifts)

    assert np.allclose(remaining_shift, [0.4, 0.1, 0.2, 0.1, -0.4])

    assert shifted_waveforms[0, 0, 9] == 1
    assert shifted_waveforms[0, 1, 8] == 1
    assert shifted_waveforms[0, 2, 12] == 1
    assert shifted_waveforms[0, 3, 7] == 1
    assert shifted_waveforms[0, 4, 14] == 1

    # 2 channel, 5 pixels, 40 samples
    waveforms = np.zeros((2, 5, 40))
    waveforms[:, :, 10] = 1
    shifts = np.array([[1.4, 2.1, -1.8, 3.1, -4.4], [1.4, 2.1, -1.8, 3.1, -4.4]])

    shifted_waveforms, remaining_shift = shift_waveforms(waveforms, shifts)

    assert np.allclose(
        remaining_shift, [[0.4, 0.1, 0.2, 0.1, -0.4], [0.4, 0.1, 0.2, 0.1, -0.4]]
    )

    assert (shifted_waveforms[:, 0, 9] == 1).all()
    assert (shifted_waveforms[:, 1, 8] == 1).all()
    assert (shifted_waveforms[:, 2, 12] == 1).all()
    assert (shifted_waveforms[:, 3, 7] == 1).all()
    assert (shifted_waveforms[:, 4, 14] == 1).all()


def test_invalid_pixels(example_event, example_subarray):
    # switching off the corrections makes it easier to test for
    # the exact value of 1.0
    config = Config(
        {
            "CameraCalibrator": {
                "image_extractor_type": "LocalPeakWindowSum",
                "apply_peak_time_shift": False,
                "LocalPeakWindowSum": {
                    "apply_integration_correction": False,
                },
            }
        }
    )
    # going to modify this
    event = deepcopy(example_event)
    tel_id = list(event.r0.tel)[0]
    camera = example_subarray.tel[tel_id].camera
    sampling_rate = camera.readout.sampling_rate.to_value(u.GHz)

    event.mon.tel[tel_id].pixel_status.flatfield_failing_pixels[:, 0] = True
    event.r1.tel[tel_id].waveform.fill(0.0)
    event.r1.tel[tel_id].waveform[:, 1:, 20] = 1.0
    event.r1.tel[tel_id].waveform[:, 0, 10] = 9999

    calibrator = CameraCalibrator(
        subarray=example_subarray,
        config=config,
    )
    calibrator(event)
    assert np.all(event.dl1.tel[tel_id].image == 1.0)
    assert np.all(event.dl1.tel[tel_id].peak_time == 20.0 / sampling_rate)

    # test we can set the invalid pixel handler to None
    config.CameraCalibrator.invalid_pixel_handler_type = None
    calibrator = CameraCalibrator(
        subarray=example_subarray,
        config=config,
    )
    calibrator(event)
    assert event.dl1.tel[tel_id].image[0] == 9999
    assert event.dl1.tel[tel_id].peak_time[0] == 10.0 / sampling_rate


def test_no_gain_selection(prod5_gamma_simtel_path):
    from ctapipe.io import SimTelEventSource

    with SimTelEventSource(prod5_gamma_simtel_path, select_gain=False) as source:
        event = next(iter(source))

    tested_n_channels = set()

    for tel_id in event.r1.tel:
        readout = source.subarray.tel[tel_id].camera.readout
        tested_n_channels.add(readout.n_channels)

        calibrator = CameraCalibrator(subarray=source.subarray)
        calibrator(event)

        image = event.dl1.tel[tel_id].image
        peak_time = event.dl1.tel[tel_id].peak_time
        assert image.ndim == 2
        assert peak_time.ndim == 2
        assert image.shape == (readout.n_channels, readout.n_pixels)
        assert peak_time.shape == (readout.n_channels, readout.n_pixels)

    assert tested_n_channels == {1, 2}
