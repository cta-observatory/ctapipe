import numpy as np
from numpy.testing import assert_almost_equal

from ctapipe.image.waveform_cleaning import (NullWaveformCleaner,
                                             CHECMWaveformCleanerAverage,
                                             CHECMWaveformCleanerLocal,
                                             BaselineWaveformCleaner)


def test_null_cleaner(example_event):
    telid = list(example_event.r0.tel)[0]
    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    cleaner = NullWaveformCleaner()
    cleaned = cleaner.apply(data_ped)

    assert (np.array_equal(data_ped, cleaned))


def test_checm_cleaner_average(example_event):
    telid = list(example_event.r0.tel)[0]
    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    cleaner = CHECMWaveformCleanerAverage()
    cleaner.apply(data_ped)


def test_checm_cleaner_local(example_event):
    telid = list(example_event.r0.tel)[0]
    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    cleaner = CHECMWaveformCleanerLocal()
    cleaner.apply(data_ped)


def test_baseline_cleaner():

    # waveform : first 20 samples = 0, second 20 samples = 10
    waveform = np.full((2, 1855, 40), 10)
    waveform[:, :, 0:20] = 0

    cleaner = BaselineWaveformCleaner()

    cleaner.baseline_start = 0
    cleaner.baseline_end = 20
    cleaned = cleaner.apply(waveform)
    assert (cleaned.mean() == 5)

    cleaner.baseline_start = 20
    cleaner.baseline_end = 40
    cleaned = cleaner.apply(waveform)
    assert (cleaned.mean() == -5)
