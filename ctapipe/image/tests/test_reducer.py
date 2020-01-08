import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.stats import norm
import astropy.units as u
from ctapipe.instrument import SubarrayDescription, TelescopeDescription
from ctapipe.image.reducer import (
    NullDataVolumeReducer,
    TailCutsDataVolumeReducer
)


@pytest.fixture(scope='module')
def camera_waveforms():
    subarray = SubarrayDescription(
        "test array",
        tel_positions={1: np.zeros(3) * u.m, 2: np.ones(3) * u.m},
        tel_descriptions={
            1: TelescopeDescription.from_name(
                optics_name="LST", camera_name="LSTCam"
            ),
            2: TelescopeDescription.from_name(
                optics_name="LST", camera_name="LSTCam"
            ),
        }
    )

    n_pixels = subarray.tel[1].camera.n_pixels
    n_samples = 30
    mid = n_samples // 2
    pulse_sigma = 6
    random = np.random.RandomState(1)

    x = np.arange(n_samples)

    # Randomize times
    t_pulse = random.uniform(mid - 10, mid + 10, n_pixels)[:, np.newaxis]

    # Create pulses
    waveforms = norm.pdf(x, t_pulse, pulse_sigma)

    # Randomize amplitudes
    waveforms *= random.uniform(100, 1000, n_pixels)[:, np.newaxis]

    return waveforms, subarray, n_samples


def test_null_data_volume_reducer(camera_waveforms):
    waveforms, _, _ = camera_waveforms
    reducer = NullDataVolumeReducer()
    reduced_waveforms_mask = reducer(waveforms)
    reduced_waveforms = waveforms.copy()
    reduced_waveforms[~reduced_waveforms_mask] = 0
    assert_array_equal(waveforms, reduced_waveforms)


def test_tailcuts_data_volume_reducer(camera_waveforms):
    waveforms, subarray, n_samples = camera_waveforms

    # create signal out of waveforms
    waveforms_signal = np.zeros_like(waveforms, dtype=np.float)

    # created set of pixels are connected in one line, not in a blob
    # Should be selected as core-pixel from Step 1) tailcuts_clean
    waveforms_signal[9] = waveforms[3]
    # waveforms[3] ~ 410.08pe after LocalPeakWindowSum

    # 10 and 8 as boundary-pixel from Step 1) tailcuts_clean
    # 6 and 5 as iteration-pixel in Step 2)
    waveforms_signal[[10, 8, 6, 5]] = waveforms[0]
    # waveforms[0] ~ 241.82pe after LocalPeakWindowSum

    # pixels from dilate at the end in Step 3)
    waveforms_signal[[0, 1, 4, 7, 11, 13, 121, 122,
                      136, 137, 257, 258, 267, 272]] = waveforms[2]
    # waveforms[2] ~ 102.59pe after LocalPeakWindowSum
    expected_waveforms = waveforms_signal.copy()

    # add some random pixels, which should not be selected
    waveforms_signal[[50, 51, 135, 138, 54, 170, 210, 400]] = waveforms[2]

    reducer = TailCutsDataVolumeReducer(subarray=subarray)
    reducer.picture_thresh = 400
    reducer.boundary_thresh = 200
    reducer.end_dilates = 1
    reducer.keep_isolated_pixels = True
    reducer.min_number_picture_neighbors = 0

    reduced_waveforms = waveforms_signal.copy()
    reduced_waveforms_mask = reducer(waveforms_signal, telid=1)
    reduced_waveforms[~reduced_waveforms_mask] = 0

    assert (reduced_waveforms != 0).sum() == (1 + 4 + 14) * n_samples
    assert_array_equal(expected_waveforms, reduced_waveforms)
