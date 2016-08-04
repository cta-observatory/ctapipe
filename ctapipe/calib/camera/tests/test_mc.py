from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path

from ..mc import set_integration_correction, calibrate_amplitude_mc, \
    integration_mc, calibrate_mc


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


def test_set_integration_correction():
    telid = 11
    event = get_test_event()
    int_corr = set_integration_correction(event, telid, get_test_parameters())
    assert int_corr == float(round(1.0497408130033212, 7))


def test_calibrate_amplitude_mc():
    telid = 11
    event = get_test_event()
    charge, window, dped = integration_mc(event, telid, get_test_parameters())
    pe = calibrate_amplitude_mc(event, charge, telid, get_test_parameters())
    assert pe[0][0] == -1.891745344400406


def test_integration_mc():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    nsamples = event.dl0.tel[telid].num_samples

    params['integrator'] = 'full_integration'
    charge, window, data_ped = integration_mc(event, telid, params)
    assert charge[0][0] == 149
    assert sum(window[0][0]) == nsamples
    assert data_ped[0][0][0] == -2.8340006510416629

    params['integrator'] = 'simple_integration'
    charge, window, data_ped = integration_mc(event, telid, params)
    assert charge[0][0] == 74
    assert sum(window[0][0]) == params['window']

    params['integrator'] = 'global_peak_integration'
    charge, window, data_ped = integration_mc(event, telid, params)
    assert charge[0][0] == 61
    assert sum(window[0][0]) == params['window']

    params['integrator'] = 'local_peak_integration'
    charge, window, data_ped = integration_mc(event, telid, params)
    assert charge[0][0] == 80
    assert sum(window[0][0]) == params['window']

    params['integrator'] = 'nb_peak_integration'
    charge, window, data_ped = integration_mc(event, telid, params)
    assert charge[0][0] == -67
    assert sum(window[0][0]) == params['window']


def test_calibrate_mc():
    telid = 11
    event = get_test_event()
    pe, window, data_ped = calibrate_mc(event, telid, get_test_parameters())
    assert pe[0][0] == -1.891745344400406
