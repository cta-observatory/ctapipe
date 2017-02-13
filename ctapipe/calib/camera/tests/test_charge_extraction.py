from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from ctapipe.io import CameraGeometry
import numpy as np
from numpy.testing import assert_almost_equal

from ctapipe.calib.camera.charge_extractors import FullIntegrator, \
    SimpleIntegrator, GlobalPeakIntegrator, LocalPeakIntegrator, \
    NeighbourPeakIntegrator, ChargeExtractorFactory


def get_test_event():
    filename = get_path('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 use_event_id=True)
    event = next(source)
    return event


def test_full_integration():
    telid = 11
    event = get_test_event()
    data = event.dl0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = FullIntegrator(None, None)
    integration = integrator.extract_charge(data_ped)
    peakpos = integrator.peakpos

    assert_almost_equal(integration[0][0], 149, 0)
    assert_almost_equal(integration[1][0], 149, 0)
    assert peakpos is None


def test_simple_integration():
    telid = 11
    event = get_test_event()
    data = event.dl0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = SimpleIntegrator(None, None)
    integration = integrator.extract_charge(data_ped)
    peakpos = integrator.peakpos

    assert_almost_equal(integration[0][0], 70, 0)
    assert_almost_equal(integration[1][0], 70, 0)
    assert peakpos is None


def test_global_peak_integration():
    telid = 11
    event = get_test_event()
    data = event.dl0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = GlobalPeakIntegrator(None, None)
    integration = integrator.extract_charge(data_ped)
    peakpos = integrator.peakpos

    assert_almost_equal(integration[0][0], 58, 0)
    assert_almost_equal(integration[1][0], 58, 0)
    assert peakpos[0][0] == 14
    assert peakpos[1][0] == 14


def test_local_peak_integration():
    telid = 11
    event = get_test_event()
    data = event.dl0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = LocalPeakIntegrator(None, None)
    integration = integrator.extract_charge(data_ped)
    peakpos = integrator.peakpos

    assert_almost_equal(integration[0][0], 76, 0)
    assert_almost_equal(integration[1][0], 76, 0)
    assert peakpos[0][0] == 13
    assert peakpos[1][0] == 13


def test_nb_peak_integration():
    telid = 11
    event = get_test_event()
    data = event.dl0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    geom = CameraGeometry.guess(*event.inst.pixel_pos[telid],
                                event.inst.optical_foclen[telid])
    nei = geom.neighbors

    integrator = NeighbourPeakIntegrator(None, None)
    integrator.neighbours = nei
    integration = integrator.extract_charge(data_ped)
    peakpos = integrator.peakpos

    print(integration[0][0])
    assert_almost_equal(integration[0][0], -64, 0)
    assert_almost_equal(integration[1][0], -64, 0)
    assert peakpos[0][0] == 20
    assert peakpos[1][0] == 20


def test_charge_extractor_factory():
    extractor_f = ChargeExtractorFactory(None, None)
    extractor_f.extractor = 'LocalPeakIntegrator'
    extractor_c = extractor_f.get_class()
    extractor = extractor_c(None, None)

    telid = 11
    event = get_test_event()
    data = event.dl0.tel[telid].adc_samples
    nsamples = data.shape[2]
    ped = event.mc.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)

    integration = extractor.extract_charge(data_ped)

    assert_almost_equal(integration[0][0], 76, 0)
