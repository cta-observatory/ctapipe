import numpy as np
from numpy.testing import assert_almost_equal

from ctapipe.image.charge_extractors import (
    ChargeExtractor,
    FullIntegrator,
    SimpleIntegrator,
    GlobalPeakIntegrator,
    LocalPeakIntegrator,
    NeighbourPeakIntegrator,
    AverageWfPeakIntegrator,
)


def test_full_integration(example_event):
    telid = list(example_event.r0.tel)[0]
    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = FullIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)


def test_simple_integration(example_event):
    telid = list(example_event.r0.tel)[0]

    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = SimpleIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)


def test_global_peak_integration(example_event):
    telid = list(example_event.r0.tel)[0]
    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = GlobalPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)


def test_local_peak_integration(example_event):
    telid = list(example_event.r0.tel)[0]
    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = LocalPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)


def test_nb_peak_integration(example_event):
    telid = list(example_event.r0.tel)[0]
    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    geom = example_event.inst.subarray.tel[telid].camera
    nei = geom.neighbor_matrix_where

    integrator = NeighbourPeakIntegrator()
    integrator.neighbours = nei
    integration, peakpos, window = integrator.extract_charge(data_ped)


def test_averagewf_peak_integration(example_event):
    telid = list(example_event.r0.tel)[0]
    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = AverageWfPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0][0], 73, 0)
    assert_almost_equal(integration[1][0], 73, 0)


def test_charge_extractor_factory(example_event):
    extractor = ChargeExtractor.from_name('LocalPeakIntegrator')

    telid = list(example_event.r0.tel)[0]
    data = example_event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = example_event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)

    extractor.extract_charge(data_ped)


def test_charge_extractor_factory_args():
    '''config is supposed to be created by a `Tool`
    '''
    from traitlets.config.loader import Config
    config = Config(
        {
            'ChargeExtractor': {
                'window_width': 20,
                'window_shift': 3,
            }
        }
    )

    local_peak_integrator = ChargeExtractor.from_name(
        'LocalPeakIntegrator',
        config=config,
    )
    assert local_peak_integrator.window_width == 20
    assert local_peak_integrator.window_shift == 3

    ChargeExtractor.from_name(
        'FullIntegrator',
        config=config,
    )
