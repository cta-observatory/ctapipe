from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from ctapipe.io import CameraGeometry
import numpy as np

from ctapipe.calib.camera.charge_extraction import FullIntegrator, \
    SimpleIntegrator, GlobalPeakIntegrator, LocalPeakIntegrator, \
    NeighbourPeakIntegrator


def get_test_event():
    filename = get_path('gamma_test.simtel.gz')
    source = hessio_event_source(filename, requested_event=409,
                                 request_event_id=True)
    event = next(source)
    return event


def test_full_integration():
    telid = 11
    event = get_test_event()
    nsamples = event.dl0.tel[telid].num_samples
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = FullIntegrator(data_ped, None)
    integration, window, peakpos = integrator.extract_charge()

    assert integration[0][0] == 149
    assert integration[1][0] == 149
    assert peakpos[0] is None
    assert peakpos[1] is None


def test_simple_integration():
    telid = 11
    event = get_test_event()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    nsamples = event.dl0.tel[telid].num_samples
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = SimpleIntegrator(data_ped, None)
    integration, window, peakpos = integrator.extract_charge()

    assert integration[0][0] == 70
    assert integration[1][0] == 70
    assert peakpos[0] is None
    assert peakpos[1] is None


def test_global_peak_integration():
    telid = 11
    event = get_test_event()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    nsamples = event.dl0.tel[telid].num_samples
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = GlobalPeakIntegrator(data_ped, None)
    integration, window, peakpos = integrator.extract_charge()

    assert integration[0][0] == 58
    assert integration[1][0] == 58
    assert peakpos[0][0] == 14
    assert peakpos[1][0] == 14


def test_local_peak_integration():
    telid = 11
    event = get_test_event()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    nsamples = event.dl0.tel[telid].num_samples
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality

    integrator = LocalPeakIntegrator(data_ped, None)
    integration, window, peakpos = integrator.extract_charge()

    assert integration[0][0] == 76
    assert integration[1][0] == 76
    assert peakpos[0][0] == 13
    assert peakpos[1][0] == 13


def test_nb_peak_integration():
    telid = 11
    event = get_test_event()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    nsamples = event.dl0.tel[telid].num_samples
    data_ped = data - np.atleast_3d(ped/nsamples)
    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                event.meta.optical_foclen[telid])
    nei = geom.neighbors

    integrator = NeighbourPeakIntegrator(data_ped, nei, None)
    integration, window, peakpos = integrator.extract_charge()

    assert integration[0][0] == -64
    assert integration[1][0] == -64
    assert peakpos[0][0] == 20
    assert peakpos[1][0] == 20