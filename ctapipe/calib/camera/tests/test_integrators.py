from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from ctapipe.io import CameraGeometry
import numpy as np

from ..integrators import integrator_switch, full_integration, \
    simple_integration, global_peak_integration, local_peak_integration, \
    nb_peak_integration


def get_test_parameters():
    parameters = {"integrator": "nb_peak_integration",
                  "window": 7,
                  "shift": 3,
                  "sigamp": [2, 4],
                  "clip_amp": 0,
                  "lwt": 0}
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
    geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                event.meta.optical_foclen[telid])

    params['integrator'] = 'full_integration'
    integration, window, peakpos = integrator_switch(data, geom, params)
    assert integration[0][0] == 3114
    assert sum(window[0][0]) == nsamples
    assert peakpos[0] is None

    params['integrator'] = 'simple_integration'
    integration, window, peakpos = integrator_switch(data, geom, params)
    assert integration[0][0] == 762
    assert sum(window[0][0]) == params['window']
    assert peakpos[0] is None

    params['integrator'] = 'global_peak_integration'
    integration, window, peakpos = integrator_switch(data, geom, params)
    assert integration[0][0] == 750
    assert sum(window[0][0]) == params['window']
    assert peakpos[0][0] == 14

    params['integrator'] = 'local_peak_integration'
    integration, window, peakpos = integrator_switch(data, geom, params)
    assert integration[0][0] == 768
    assert sum(window[0][0]) == params['window']
    assert peakpos[0][0] == 13

    params['integrator'] = 'nb_peak_integration'
    integration, window, peakpos = integrator_switch(data, geom, params)
    assert integration[0][0] == 628
    assert sum(window[0][0]) == params['window']
    assert peakpos[0][0] == 20


def test_full_integration():
    telid = 11
    event = get_test_event()
    nsamples = event.dl0.tel[telid].num_samples
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))

    data = np.array([data[0], data[0]])  # Test 2 channel functionality
    integration, window, peakpos = full_integration(data)
    assert integration[0][0] == 3114
    assert integration[1][0] == 3114
    assert sum(window[0][0]) == nsamples
    assert sum(window[1][0]) == nsamples
    assert peakpos[0] is None
    assert peakpos[1] is None


def test_simple_integration():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))

    data = np.array([data[0], data[0]])  # Test 2 channel functionality
    integration, window, peakpos = simple_integration(data, params)
    assert integration[0][0] == 762
    assert integration[1][0] == 762
    assert sum(window[0][0]) == params['window']
    assert sum(window[1][0]) == params['window']
    assert peakpos[0] is None
    assert peakpos[1] is None


def test_global_peak_integration():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))

    data = np.array([data[0], data[0]])  # Test 2 channel functionality
    integration, window, peakpos = global_peak_integration(data, params)
    assert integration[0][0] == 750
    assert integration[1][0] == 750
    assert sum(window[0][0]) == params['window']
    assert sum(window[1][0]) == params['window']
    assert peakpos[0][0] == 14
    assert peakpos[1][0] == 14


def test_local_peak_integration():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))

    data = np.array([data[0], data[0]])  # Test 2 channel functionality
    integration, window, peakpos = local_peak_integration(data, params)
    assert integration[0][0] == 768
    assert integration[1][0] == 768
    assert sum(window[0][0]) == params['window']
    assert sum(window[1][0]) == params['window']
    assert peakpos[0][0] == 13
    assert peakpos[1][0] == 13


def test_nb_peak_integration():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
    geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                event.meta.optical_foclen[telid])

    data = np.array([data[0], data[0]])  # Test 2 channel functionality
    integration, window, peakpos = nb_peak_integration(data, geom, params)
    assert integration[0][0] == 628
    assert integration[1][0] == 628
    assert sum(window[0][0]) == params['window']
    assert sum(window[1][0]) == params['window']
    assert peakpos[0][0] == 20
    assert peakpos[1][0] == 20
