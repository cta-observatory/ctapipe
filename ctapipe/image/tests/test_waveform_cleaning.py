import numpy as np
from numpy.testing import assert_almost_equal

from ctapipe.image.waveform_cleaning import NullWaveformCleaner, \
    CHECMWaveformCleanerAverage, CHECMWaveformCleanerLocal
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset


def get_test_event():
    filename = get_dataset('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 use_event_id=True)
    event = next(source)
    return event


def test_null_cleaner():
    telid = 11
    event = get_test_event()
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    cleaner = NullWaveformCleaner(None, None)
    cleaned = cleaner.apply(data_ped)

    assert(np.array_equal(data_ped, cleaned))


def test_checm_cleaner_average():
    telid = 11
    event = get_test_event()
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    cleaner = CHECMWaveformCleanerAverage(None, None)
    cleaned = cleaner.apply(data_ped)

    assert_almost_equal(data_ped[0, 0, 0], -2.8, 1)
    assert_almost_equal(cleaned[0, 0, 0], -6.4, 1)


def test_checm_cleaner_local():
    telid = 11
    event = get_test_event()
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    cleaner = CHECMWaveformCleanerLocal(None, None)
    cleaned = cleaner.apply(data_ped)

    assert_almost_equal(data_ped[0, 0, 0], -2.8, 1)
    assert_almost_equal(cleaned[0, 0, 0], -15.9, 1)
