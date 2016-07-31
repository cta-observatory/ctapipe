from pyhessio import *
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from ctapipe.calib.camera.mc import *
from ctapipe.instrument import InstrumentDescription as ID


def get_test_parameters():
    parameters = {"integrator": "nb_peak_integration",
                  "window": 7,
                  "shift": 3,
                  "sigamp": [2, 4],
                  "clip_amp": 0,
                  "lwt": 0}
    return parameters


def get_test_event():
    filename = get_path(
        'gamma_test.simtel.gz')
    for event in hessio_event_source(filename):
        if event.dl0.event_id == 409:
            return event

# def get_camera_info():
#     filename = get_path(
#         'gamma_test.simtel.gz')
#     tel, cam, opt = ID.load(filename)
#     return cam


def test_set_integration_correction():
    telid = 11
    event = get_test_event()

    assert set_integration_correction(event,
        telid, get_test_parameters()) == float(round(1.0497408130033212,7))


def test_full_integration_mc():
    telid = 11
    int_adc_pix = full_integration_mc(get_test_event(), telid)
    assert int_adc_pix[0][0] == 149


def test_simple_integration_mc():
    telid = 11
    int_adc_pix = simple_integration_mc(get_test_event(), telid,
                                        get_test_parameters())
    assert int_adc_pix[0][0] == 74


def test_global_peak_integration_mc():
    telid = 11
    int_adc_pix = global_peak_integration_mc(get_test_event(), telid,
                                             get_test_parameters())
    assert int_adc_pix[0][0] == 61


def test_local_peak_integration_mc():
    telid = 11
    int_adc_pix = local_peak_integration_mc(get_test_event(), telid,
                                            get_test_parameters())
    assert int_adc_pix[0][0] == 156


def test_nb_peak_integration_mc():
    telid = 11
    int_adc_pix = nb_peak_integration_mc(get_test_event(), telid,
                                         get_test_parameters())
    assert int_adc_pix[0][0] == 156


def test_pixel_integration_mc():
    telid = 11
    event = get_test_event()
    int_adc_pix = pixel_integration_mc(event, telid, get_test_parameters())
    assert int_adc_pix[0][0] == 156


def test_calibrate_amplitude_mc():
    telid = 11
    event = get_test_event()
    int_adc_pix = pixel_integration_mc(event, telid, get_test_parameters())
    pe_pix = calibrate_amplitude_mc(event, int_adc_pix, telid, get_test_parameters())
    assert pe_pix[0][0] == 5.0270585238933565