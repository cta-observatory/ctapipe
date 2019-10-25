import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.stats import norm
from ctapipe.image.extractor import NeighborPeakWindowSum
from ctapipe.instrument import CameraGeometry
from ctapipe.image.reducer import (
    NullDataVolumeReducer,
    TailCutsDataVolumeReducer
)


def test_null_data_volume_reducer():
    waveforms = np.random.uniform(0, 1, (2048, 96))
    reducer = NullDataVolumeReducer()
    reduced_waveforms = reducer(waveforms)
    assert_array_equal(waveforms, reduced_waveforms)


def test_tailcuts_data_volume_reducer():
    # Create waveforms like 'test_extractor'
    camera = CameraGeometry.from_name("LSTCam")
    n_pixels = camera.n_pixels
    n_samples = 30
    mid = n_samples // 2
    pulse_sigma = 6
    random = np.random.RandomState(1)

    x = np.arange(n_samples)

    # Randomize times
    t_pulse = random.uniform(mid - 10, mid + 10, n_pixels)[:, np.newaxis]

    # Create pulses
    waveforms_values = norm.pdf(x, t_pulse, pulse_sigma)

    # Randomize amplitudes
    waveforms_values *= random.uniform(100, 1000, n_pixels)[:, np.newaxis]

    # to test the used image_extractor in TailcutsDataVolumeReducer
    image_extractor = NeighborPeakWindowSum()
    image_extractor.neighbors = camera.neighbor_matrix_where
    charge, pulse_time = image_extractor(waveforms_values)

    # create image
    waveforms = np.zeros_like(waveforms_values, dtype=np.float)

    # created set of pixels are connected in one line, not in a blob
    # Should be selected as core-pixel from Step 1) tailcuts_clean
    waveforms[9] = waveforms_values[3]

    # 10 and 8 as boundary-pixel from Step 1) tailcuts_clean
    # 6 and 5 as iteration-pixel in Step 2)
    waveforms[[10, 8, 6, 5]] = waveforms_values[0]

    # pixels from dilate at the end in Step 3)
    waveforms[[0, 1, 4, 7, 11, 13, 121, 122,
               136, 137, 257, 258, 267, 272]] = waveforms_values[2]
    expected_waveforms = waveforms.copy()

    # add some random pixels, which should not be selected
    waveforms[[50, 51, 52, 53, 54, 170, 210, 400]] = waveforms_values[2]

    reducer = TailCutsDataVolumeReducer()
    reducer.camera_geom = camera
    reducer.picture_thresh = 400
    reducer.boundary_thresh = 100
    reducer.end_dilates = 1
    reducer.keep_isolated_pixels = True
    reducer.min_number_picture_neighbors = 0
    reduced_waveforms = reducer(waveforms)

    assert (reduced_waveforms != 0).sum() == 19 * n_samples
    assert_allclose(charge[0], 191.6570529, rtol=1e-3)
    assert_allclose(charge[2], 65.61672762, rtol=1e-3)
    assert_allclose(charge[3], 404.559261, rtol=1e-3)
    assert_array_equal(expected_waveforms, reduced_waveforms)
