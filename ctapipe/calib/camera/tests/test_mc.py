from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path

from ..mc import set_integration_correction, calibrate_amplitude_mc, \
    integration_mc, calibrate_mc


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


def test_set_integration_correction():
    telid = 11
    event = get_test_event()
    int_corr = set_integration_correction(event, telid, get_test_parameters())
    assert int_corr == float(round(1.0497408130033212, 7))


def test_calibrate_amplitude_mc():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    charge, window, dped, peakpos = integration_mc(event, telid, params)
    pe = calibrate_amplitude_mc(event, charge, telid, get_test_parameters())
    assert pe[0][0] == -1.891745344400406


def test_integration_mc():
    telid = 11
    event = get_test_event()
    params = get_test_parameters()
    nsamples = event.dl0.tel[telid].num_samples

    params['integrator'] = 'full_integration'
    charge, window, data_ped, peakpos = integration_mc(event, telid, params)
    assert charge[0][0] == 149
    assert sum(window[0][0]) == nsamples
    assert data_ped[0][0][0] == -2.8340006510416629

    params['integrator'] = 'simple_integration'
    charge, window, data_ped, peakpos = integration_mc(event, telid, params)
    assert charge[0][0] == 70
    assert sum(window[0][0]) == params['integration_window'][0]

    params['integrator'] = 'global_peak_integration'
    charge, window, data_ped, peakpos = integration_mc(event, telid, params)
    assert charge[0][0] == 61
    assert sum(window[0][0]) == params['integration_window'][0]

    params['integrator'] = 'local_peak_integration'
    charge, window, data_ped, peakpos = integration_mc(event, telid, params)
    assert charge[0][0] == 80
    assert sum(window[0][0]) == params['integration_window'][0]

    params['integrator'] = 'nb_peak_integration'
    charge, window, data_ped, peakpos = integration_mc(event, telid, params)
    assert charge[0][0] == -67
    assert sum(window[0][0]) == params['integration_window'][0]


def test_calibrate_mc():
    telid = 11
    event = get_test_event()
    pe, window, data_ped, peakpos = calibrate_mc(event, telid,
                                                 get_test_parameters())
    assert round(pe[0], 5) == -1.89175
