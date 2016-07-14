from pyhessio import *
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_path
from ctapipe.calib.camera.mc import *
from ctapipe.instrument import InstrumentDescription as ID


def get_test_parameters():
    parameters = {"integrator": "nb_peak_integration",
                  "nsum": 7,
                  "nskip": 3,
                  "sigamp": [2, 4],
                  "clip_amp": 0,
                  "lwt": 0}
    return parameters


def get_camera_info():
    filename = get_path(
        'gamma_test.simtel.gz')
    tel, cam, opt = ID.load(filename)
    return cam


def get_test_event():
    filename = get_path(
        'gamma_test.simtel.gz')
    for event in hessio_event_source(filename):
        if event.dl0.event_id == 409:
            return event


def test_set_integration_correction():
    telid = 11
    event = get_test_event()

    assert set_integration_correction(
        telid, get_test_parameters()) == float(round(1.0497408130033212, 7))


def test_full_integration_mc():
    telid = 11
    int_adc_pix, peak_adc_pix = full_integration_mc(
        get_test_event(), get_pedestal(telid), telid)
    assert int_adc_pix[0][0] == 148
    assert peak_adc_pix is None


def test_simple_integration_mc():
    telid = 11
    int_adc_pix, peak_adc_pix = simple_integration_mc(
        get_test_event(), get_pedestal(telid), telid, get_test_parameters())
    assert int_adc_pix[0][0] == int(70)
    assert peak_adc_pix is None


def test_global_peak_integration_mc():
    telid = 11
    int_adc_pix, peak_adc_pix = global_peak_integration_mc(
        get_test_event(), get_pedestal(telid), telid, get_test_parameters())
    assert int_adc_pix[0][0] == int(77) 
    assert peak_adc_pix[0] == int(13)


def test_local_peak_integration_mc():
    telid = 11
    int_adc_pix, peak_adc_pix = local_peak_integration_mc(
        get_test_event(), get_pedestal(telid), telid, get_test_parameters())
    assert int_adc_pix[0][0] == int(77)
    assert peak_adc_pix[0] == int(13)


def test_nb_peak_integration_mc():
    telid = 11
    cam = get_camera_info()
    int_adc_pix, peak_adc_pix = nb_peak_integration_mc(
        get_test_event(), cam, get_pedestal(telid),
        telid, get_test_parameters())
    assert int_adc_pix[0][0] == int(-61)
    assert peak_adc_pix[0] == int(20)


def test_pixel_integration_mc():
    telid = 11
    cam = get_camera_info()
    int_adc_pix, peak_adc_pix = pixel_integration_mc(
        get_test_event(), cam, get_pedestal(telid),
        telid, get_test_parameters())

    assert int_adc_pix[0][0] == int(-61)
    assert peak_adc_pix[0] == int(20)


def test_calibrate_amplitude_mc():
    telid = 11
    cam = get_camera_info()
    int_adc_pix, peak_adc_pix = pixel_integration_mc(
        get_test_event(), cam, get_pedestal(telid),
        telid, get_test_parameters())
    calib = get_calibration(telid)
    pe_pix = calibrate_amplitude_mc(
        int_adc_pix, calib, telid, get_test_parameters())

    assert pe_pix[0] == float(-1.7223353135585786)
