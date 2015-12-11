from pyhessio import *
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from ctapipe.calib.camera.mc import *


def get_test_parameters():
    parameters = {"integrator": "nb_peak_integration",
                  "nsum": 7,
                  "nskip": 3,
                  "sigamp": [2, 4],
                  "clip_amp": 0,
                  "lwt": 0}
    return parameters


def get_test_event():
    filename = get_path(
        'gamma_20deg_0deg_run31964___cta-prod2_desert-1640m-Aar.simtel.gz')
    for event in hessio_event_source(filename):
        if event.dl0.event_id == 409:
            return event


def test_set_integration_correction():
    telid = 11
    event = get_test_event()

    assert set_integration_correction(
        telid, get_test_parameters()) == 1.059373


def test_pixel_integration_mc():
    telid = 11
    event = get_test_event()
    ped = get_pedestal(telid)
    int_adc_pix, peak_adc_pix = pixel_integration_mc(
        event, ped, telid, get_test_parameters())

    assert int_adc_pix[0][0] == -68.0
    assert peak_adc_pix[0][0] == 20


def test_calibrate_amplitude_mc():
    telid = 11
    event = get_test_event()
    ped = get_pedestal(telid)
    int_adc_pix, peak_adc_pix = pixel_integration_mc(
        event, ped, telid, get_test_parameters())
    calib = get_calibration(telid)
    pe_pix = calibrate_amplitude_mc(
        int_adc_pix, calib, telid, get_test_parameters())

    assert pe_pix[0] == -1.92
