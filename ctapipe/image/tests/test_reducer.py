import pytest
import numpy as np
from numpy.testing import assert_array_equal
import astropy.units as u
from traitlets.config import Config
from ctapipe.instrument import SubarrayDescription, TelescopeDescription
from ctapipe.image.reducer import NullDataVolumeReducer, TailCutsDataVolumeReducer


@pytest.fixture(scope="module")
def subarray_lst():
    telid = 1
    subarray = SubarrayDescription(
        "test array lst",
        tel_positions={1: np.zeros(3) * u.m, 2: np.ones(3) * u.m},
        tel_descriptions={
            1: TelescopeDescription.from_name(optics_name="LST", camera_name="LSTCam"),
            2: TelescopeDescription.from_name(optics_name="LST", camera_name="LSTCam"),
        },
    )

    n_pixels = subarray.tel[telid].camera.geometry.n_pixels
    n_samples = 30
    selected_gain_channel = np.zeros(n_pixels, dtype=np.int)

    return subarray, telid, selected_gain_channel, n_pixels, n_samples


def test_null_data_volume_reducer(subarray_lst):
    subarray, _, _, _, _ = subarray_lst
    waveforms = np.random.uniform(0, 1, (2048, 96))
    reducer = NullDataVolumeReducer(subarray=subarray)
    reduced_waveforms_mask = reducer(waveforms)
    reduced_waveforms = waveforms.copy()
    reduced_waveforms[~reduced_waveforms_mask] = 0
    assert_array_equal(waveforms, reduced_waveforms)


def test_tailcuts_data_volume_reducer(subarray_lst):
    subarray, telid, selected_gain_channel, n_pixels, n_samples = subarray_lst

    # create signal
    waveforms_signal = np.zeros((n_pixels, n_samples), dtype=np.float)

    # Should be selected as core-pixel from Step 1) tailcuts_clean
    waveforms_signal[9] = 100

    # 10 and 8 as boundary-pixel from Step 1) tailcuts_clean
    # 6 and 5 as iteration-pixel in Step 2)
    waveforms_signal[[10, 8, 6, 5]] = 50

    # pixels from dilate at the end in Step 3)
    waveforms_signal[[0, 1, 4, 7, 11, 13, 121, 122, 136, 137, 257, 258, 267, 272]] = 25

    expected_waveforms = waveforms_signal.copy()

    # add some random pixels, which should not be selected
    waveforms_signal[[50, 51, 135, 138, 54, 170, 210, 400]] = 25

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
                "n_end_dilates": 1,
                "do_boundary_dilation": True,
            }
        }
    )
    reducer = TailCutsDataVolumeReducer(config=reduction_param, subarray=subarray)
    reduced_waveforms = waveforms_signal.copy()
    reduced_waveforms_mask = reducer(
        waveforms_signal, telid=telid, selected_gain_channel=selected_gain_channel
    )
    reduced_waveforms[~reduced_waveforms_mask] = 0

    assert (reduced_waveforms != 0).sum() == (1 + 4 + 14) * n_samples
    assert_array_equal(expected_waveforms, reduced_waveforms)
