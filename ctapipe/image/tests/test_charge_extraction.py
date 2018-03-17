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

def dummy_calib(event, telid=11):
    data = event.r0.tel[telid].waveform
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped / nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    return data_ped[0]


def test_full_integration(test_event):
    data_ped = dummy_calib(deepcopy(test_event))

    integrator = FullIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0], 149, 0)

    assert peakpos[0] == 0



def test_simple_integration(test_event):
    data_ped = dummy_calib(deepcopy(test_event), telid=11)

    integrator = SimpleIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0], 74, 0)
    assert peakpos[0] == 0



def test_global_peak_integration(test_event):
    data_ped = dummy_calib(deepcopy(test_event), telid=11)

    integrator = GlobalPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0], 58, 0)
    assert peakpos[0] == 14


def test_local_peak_integration(test_event):
    data_ped = dummy_calib(deepcopy(test_event), telid=11)

    integrator = LocalPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0], 76, 0)
    assert peakpos[0] == 13


def test_nb_peak_integration(test_event):
    data_ped = dummy_calib(deepcopy(test_event), telid=11)

    geom = test_event.inst.subarray.tel[11].camera
    nei = geom.neighbor_matrix_where

    integrator = NeighbourPeakIntegrator()
    integrator.neighbours = nei
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0], -64, 0)
    assert peakpos[0] == 20



def test_averagewf_peak_integration(test_event):
    data_ped = dummy_calib(deepcopy(test_event), telid=11)

    integrator = AverageWfPeakIntegrator()
    integration, peakpos, window = integrator.extract_charge(data_ped)

    assert_almost_equal(integration[0], 73, 0)
    assert peakpos[0] == 10



def test_charge_extractor_factory(test_event):
    extractor = ChargeExtractorFactory.produce(

        product='LocalPeakIntegrator'
    )

    data_ped = dummy_calib(deepcopy(test_event), telid=11)

    integration, peakpos, window = extractor.extract_charge(data_ped)

    assert_almost_equal(integration[0], 76, 0)
