"""
Tests for CameraCalibrator and related functions
"""
import numpy as np
import pytest
from scipy.stats import norm
from traitlets.config.configurable import Config

from ctapipe.calib.camera.calibrator import CameraCalibrator
from ctapipe.image.extractor import (
    NeighborPeakWindowSum,
    LocalPeakWindowSum,
    FullWaveformSum,
)
from ctapipe.image.reducer import NullDataVolumeReducer, TailCutsDataVolumeReducer
from ctapipe.instrument import CameraGeometry
from ctapipe.containers import DataContainer


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
    event = DataContainer()
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
    event = DataContainer()
    event.dl1.tel[telid].image = np.full(2048, 2)
    with pytest.warns(UserWarning):
        calibrator(event)
    assert (event.dl1.tel[telid].image == 2).all()


def test_dl1_charge_calib(example_subarray):
    camera = CameraGeometry.from_name("CHEC")
    n_pixels = camera.n_pixels
    n_samples = 96
    mid = n_samples // 2
    pulse_sigma = 6
    random = np.random.RandomState(1)
    x = np.arange(n_samples)

    # Randomize times and create pulses
    time_offset = random.uniform(mid - 10, mid + 10, n_pixels)[:, np.newaxis]
    y = norm.pdf(x, time_offset, pulse_sigma).astype("float32")

    # Define absolute calibration coefficients
    absolute = random.uniform(100, 1000, n_pixels).astype("float32")
    y *= absolute[:, np.newaxis]

    # Define relative coefficients
    relative = random.normal(1, 0.01, n_pixels)
    y /= relative[:, np.newaxis]

    # Define pedestal
    pedestal = random.uniform(-4, 4, n_pixels)
    y += pedestal[:, np.newaxis]

    event = DataContainer()
    telid = list(example_subarray.tel.keys())[0]
    event.dl0.tel[telid].waveform = y

    # Test default
    calibrator = CameraCalibrator(
        subarray=example_subarray,
        image_extractor=FullWaveformSum(subarray=example_subarray),
    )
    calibrator(event)
    np.testing.assert_allclose(event.dl1.tel[telid].image, y.sum(1), rtol=1e-4)

    event.calibration.tel[telid].dl1.time_shift = time_offset
    event.calibration.tel[telid].dl1.pedestal_offset = pedestal * n_samples
    event.calibration.tel[telid].dl1.absolute_factor = absolute
    event.calibration.tel[telid].dl1.relative_factor = relative

    # Test without need for timing corrections
    calibrator = CameraCalibrator(
        subarray=example_subarray,
        image_extractor=FullWaveformSum(subarray=example_subarray),
    )
    calibrator(event)
    np.testing.assert_allclose(event.dl1.tel[telid].image, 1, rtol=1e-5)

    # TODO: Test with timing corrections
