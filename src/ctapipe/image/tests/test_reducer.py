import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from traitlets.config import Config

from ctapipe.image.reducer import NullDataVolumeReducer, TailCutsDataVolumeReducer
from ctapipe.instrument import SubarrayDescription


@pytest.fixture(scope="module")
def subarray_lst(prod3_lst, reference_location):
    tel_id = 1
    subarray = SubarrayDescription(
        "test array lst",
        tel_positions={1: np.zeros(3) * u.m, 2: np.ones(3) * u.m},
        tel_descriptions={
            1: prod3_lst,
            2: prod3_lst,
        },
        reference_location=reference_location,
    )

    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
    n_samples = 30
    selected_gain_channel = np.zeros(n_pixels, dtype=np.int16)

    return subarray, tel_id, selected_gain_channel, n_pixels, n_samples


def test_null_data_volume_reducer(subarray_lst):
    subarray, _, _, _, _ = subarray_lst
    rng = np.random.default_rng(0)
    waveforms = rng.uniform(0, 1, (1, 2048, 96))
    reducer = NullDataVolumeReducer(subarray=subarray)
    reduced_waveforms_mask = reducer(waveforms)
    reduced_waveforms = waveforms.copy()
    reduced_waveforms[:, ~reduced_waveforms_mask] = 0
    assert_array_equal(waveforms, reduced_waveforms)


def test_tailcuts_data_volume_reducer(subarray_lst):
    subarray, tel_id, selected_gain_channel, n_pixels, n_samples = subarray_lst

    # create signal
    n_channels = 1
    waveforms_signal = np.zeros((n_channels, n_pixels, n_samples), dtype=np.float64)

    # Should be selected as core-pixel from Step 1) tailcuts_clean
    waveforms_signal[0][9] = 100

    # 10 and 8 as boundary-pixel from Step 1) tailcuts_clean
    # 6 and 5 as iteration-pixel in Step 2)
    waveforms_signal[0][[10, 8, 6, 5]] = 50

    # pixels from dilate at the end in Step 3)
    waveforms_signal[0][
        [0, 1, 4, 7, 11, 13, 121, 122, 136, 137, 257, 258, 267, 272]
    ] = 25

    expected_waveforms = waveforms_signal.copy()

    # add some random pixels, which should not be selected
    waveforms_signal[0][[50, 51, 135, 138, 54, 170, 210, 400]] = 25

    # Reduction parameters
    reduction_param = Config(
        {
            "TailCutsDataVolumeReducer": {
                "TailcutsImageCleaner": {
                    "picture_threshold_pe": 700.0,
                    "boundary_threshold_pe": 350.0,
                    "min_picture_neighbors": 0,
                    "keep_isolated_pixels": True,
                },
                "image_extractor_type": "NeighborPeakWindowSum",
                "NeighborPeakWindowSum": {
                    "apply_integration_correction": False,
                    "window_shift": 0,
                },
                "n_end_dilates": 1,
                "do_boundary_dilation": True,
            }
        }
    )
    reducer = TailCutsDataVolumeReducer(config=reduction_param, subarray=subarray)
    reduced_waveforms = waveforms_signal.copy()
    reduced_waveforms_mask = reducer(
        waveforms_signal, tel_id=tel_id, selected_gain_channel=selected_gain_channel
    )
    reduced_waveforms[:, ~reduced_waveforms_mask] = 0

    assert (reduced_waveforms != 0).sum() == (1 + 4 + 14) * n_samples
    assert_array_equal(expected_waveforms, reduced_waveforms)
