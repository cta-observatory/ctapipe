from copy import deepcopy

import numpy as np
from numpy.testing import assert_almost_equal

from ctapipe.image.charge_extractors import (FullIntegrator,
                                             SimpleIntegrator,
                                             GlobalPeakIntegrator,
                                             LocalPeakIntegrator,
                                             NeighbourPeakIntegrator,
                                             ChargeExtractorFactory,
                                             AverageWfPeakIntegrator)


def test_full_integration(test_event):
    telid = 11
    event = deepcopy(test_event)
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = FullIntegrator(None, None)
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0][0], 149, 0)
    assert_almost_equal(integration[1][0], 149, 0)
    assert peakpos[0][0] == 0
    assert peakpos[1][0] == 0


def test_simple_integration(test_event):
    telid = 11
    event = deepcopy(test_event)
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = SimpleIntegrator(None, None)
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0][0], 74, 0)
    assert_almost_equal(integration[1][0], 74, 0)
    assert peakpos[0][0] == 0
    assert peakpos[1][0] == 0


def test_global_peak_integration(test_event):
    telid = 11
    event = deepcopy(test_event)
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = GlobalPeakIntegrator(None, None)
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0][0], 58, 0)
    assert_almost_equal(integration[1][0], 58, 0)
    assert peakpos[0][0] == 14
    assert peakpos[1][0] == 14


def test_local_peak_integration(test_event):
    telid = 11
    event = deepcopy(test_event)
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = LocalPeakIntegrator(None, None)
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0][0], 76, 0)
    assert_almost_equal(integration[1][0], 76, 0)
    assert peakpos[0][0] == 13
    assert peakpos[1][0] == 13


def test_nb_peak_integration(test_event):
    telid = 11
    event = deepcopy(test_event)
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    geom = event.inst.subarray.tel[telid].camera
    nei = geom.neighbor_matrix_where

    integrator = NeighbourPeakIntegrator(None, None)
    integrator.neighbours = nei
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0][0], -64, 0)
    assert_almost_equal(integration[1][0], -64, 0)
    assert peakpos[0][0] == 20
    assert peakpos[1][0] == 20


def test_averagewf_peak_integration(test_event):
    telid = 11
    event = deepcopy(test_event)
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = AverageWfPeakIntegrator(None, None)
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0][0], 73, 0)
    assert_almost_equal(integration[1][0], 73, 0)
    assert peakpos[0][0] == 10
    assert peakpos[1][0] == 10


def test_charge_extractor_factory(test_event):
    extractor = ChargeExtractorFactory.produce(
        None, None,
        product='LocalPeakIntegrator'
    )

    telid = 11
    event = deepcopy(test_event)
    data = event.r0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)

    integration, peakpos, window = extractor.extract_charge(data_ped)

    assert_almost_equal(integration[0][0], 76, 0)
