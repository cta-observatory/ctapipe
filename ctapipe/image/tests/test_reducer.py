import numpy as np
from numpy.testing import assert_array_equal
from ctapipe.image.reducer import NullDataVolumeReducer


def test_null_data_volume_reducer():
    waveforms = np.random.uniform(0, 1, (2048, 96))
    reducer = NullDataVolumeReducer()
    reduced_waveforms = reducer(waveforms)
    assert_array_equal(waveforms, reduced_waveforms)
