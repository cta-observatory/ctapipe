from ..calibrators import calibrate_event, calibrate_source
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path


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


def test_calibrate_event():
    telid = 11
    event = get_test_event()
    calibrated = calibrate_event(event, get_test_parameters())
    pe = calibrated.dl1.tel[telid].pe_charge
    assert round(pe[0], 5) == -1.89175


def test_calibrate_source():
    telid = 38
    filename = get_path('gamma_test.simtel.gz')
    source = hessio_event_source(filename)
    c_source = calibrate_source(source, get_test_parameters())
    for event in c_source:
        if event.dl1.event_id == 408:
            pe = event.dl1.tel[telid].pe_charge
            assert round(pe[0], 5) == 1.86419
            break
