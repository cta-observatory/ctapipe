from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from ctapipe.io import CameraGeometry
import numpy as np

from ..integrators import integrator_switch, full_integration, \
    simple_integration, global_peak_integration, local_peak_integration, \
    nb_peak_integration


def get_test_parameters():
    parameters = {"integrator": "nb_peak_integration",
                  "integration_window": [7, 3],
                  "integration_sigamp": [2, 4],
                  "integration_lwt": 0}
    return parameters


def get_test_event():
    filename = get_path('gamma_test.simtel.gz')
    for event in hessio_event_source(filename):
        if event.dl0.event_id == 409:
            return event


def test_integrator_switch():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    nsamples = event.dl0.tel[telid].num_samples
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)
    geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                event.meta.optical_foclen[telid])

    params['integrator'] = 'full_integration'
    integration, window, peakpos = integrator_switch(data_ped, geom, params)
    assert integration[0][0] == 149
    assert sum(window[0][0]) == nsamples
    assert peakpos[0] is None

    params['integrator'] = 'simple_integration'
    integration, window, peakpos = integrator_switch(data_ped, geom, params)
    assert integration[0][0] == 70
    assert sum(window[0][0]) == params['integration_window'][0]
    assert peakpos[0] is None

    params['integrator'] = 'global_peak_integration'
    integration, window, peakpos = integrator_switch(data_ped, geom, params)
    assert integration[0][0] == 58
    assert sum(window[0][0]) == params['integration_window'][0]
    assert peakpos[0][0] == 14

    params['integrator'] = 'local_peak_integration'
    integration, window, peakpos = integrator_switch(data_ped, geom, params)
    assert integration[0][0] == 76
    assert sum(window[0][0]) == params['integration_window'][0]
    assert peakpos[0][0] == 13

    params['integrator'] = 'nb_peak_integration'
    integration, window, peakpos = integrator_switch(data_ped, geom, params)
    assert integration[0][0] == -64
    assert sum(window[0][0]) == params['integration_window'][0]
    assert peakpos[0][0] == 20


def test_full_integration():
    telid = 11
    event = get_test_event()
    nsamples = event.dl0.tel[telid].num_samples
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    data_ped = data - np.atleast_3d(ped/nsamples)

    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    integration, window, peakpos = full_integration(data_ped)
    assert integration[0][0] == 149
    assert integration[1][0] == 149
    assert sum(window[0][0]) == nsamples
    assert sum(window[1][0]) == nsamples
    assert peakpos[0] is None
    assert peakpos[1] is None


def test_simple_integration():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    nsamples = event.dl0.tel[telid].num_samples
    data_ped = data - np.atleast_3d(ped/nsamples)

    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    integration, window, peakpos = simple_integration(data_ped, params)
    assert integration[0][0] == 70
    assert integration[1][0] == 70
    assert sum(window[0][0]) == params['integration_window'][0]
    assert sum(window[1][0]) == params['integration_window'][0]
    assert peakpos[0] is None
    assert peakpos[1] is None


def test_global_peak_integration():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    nsamples = event.dl0.tel[telid].num_samples
    data_ped = data - np.atleast_3d(ped/nsamples)

    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    integration, window, peakpos = global_peak_integration(data_ped, params)
    assert integration[0][0] == 58
    assert integration[1][0] == 58
    assert sum(window[0][0]) == params['integration_window'][0]
    assert sum(window[1][0]) == params['integration_window'][0]
    assert peakpos[0][0] == 14
    assert peakpos[1][0] == 14


def test_local_peak_integration():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    nsamples = event.dl0.tel[telid].num_samples
    data_ped = data - np.atleast_3d(ped/nsamples)

    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    integration, window, peakpos = local_peak_integration(data_ped, params)
    assert integration[0][0] == 76
    assert integration[1][0] == 76
    assert sum(window[0][0]) == params['integration_window'][0]
    assert sum(window[1][0]) == params['integration_window'][0]
    assert peakpos[0][0] == 13
    assert peakpos[1][0] == 13


def test_nb_peak_integration():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    ped = event.dl0.tel[telid].pedestal
    nsamples = event.dl0.tel[telid].num_samples
    data_ped = data - np.atleast_3d(ped/nsamples)
    geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                event.meta.optical_foclen[telid])

    data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
    integration, window, peakpos = nb_peak_integration(data_ped, geom, params)
    assert integration[0][0] == -64
    assert integration[1][0] == -64
    assert sum(window[0][0]) == params['integration_window'][0]
    assert sum(window[1][0]) == params['integration_window'][0]
    assert peakpos[0][0] == 20
    assert peakpos[1][0] == 20
