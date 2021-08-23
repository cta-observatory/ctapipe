from astropy import units as u
from pytest import raises


def test_guessing():
    from ctapipe.instrument import guess_telescope

    guess = guess_telescope(2048, 2.28)
    assert guess.type == "SST"
    assert guess.name == "GCT"

    guess = guess_telescope(2048, 2.28 * u.m)
    assert guess.type == "SST"
    assert guess.name == "GCT"

    with raises(ValueError):
        guess = guess_telescope(100, 2.28 * u.m)

    foclen = 16 * u.m
    n_pixels = 1764
    guess = guess_telescope(n_pixels, foclen)

    assert guess.camera_name == "FlashCam"
    assert guess.type == "MST"


def test_unknown_telescope():
    from ctapipe.instrument.guess import unknown_telescope

    tel = unknown_telescope(486, 1855)
    assert tel.type == "LST"
    assert tel.name == "UNKNOWN-486M2"
    assert tel.camera_name == "UNKNOWN-1855PX"

    tel = unknown_telescope(10, 2048)
    assert tel.type == "SST"
    assert tel.name == "UNKNOWN-10M2"
    assert tel.camera_name == "UNKNOWN-2048PX"

    tel = unknown_telescope(100 * u.m ** 2, 2048)
    assert tel.type == "MST"
    assert tel.name == "UNKNOWN-100M2"
    assert tel.camera_name == "UNKNOWN-2048PX"
