from astropy import units as u
from pytest import raises


def test_guessing():
    from ctapipe.instrument import guess_telescope

    guess = guess_telescope(2048, 2.28)
    assert guess.type == 'SST'
    assert guess.name == 'GCT'

    guess = guess_telescope(2048, 2.28 * u.m)
    assert guess.type == 'SST'
    assert guess.name == 'GCT'

    with raises(ValueError):
        guess = guess_telescope(100, 2.28 * u.m)

    foclen = 16 * u.m
    n_pixels = 1764
    guess = guess_telescope(n_pixels, foclen)

    assert guess.camera_name == 'FlashCam'
    assert guess.type == 'MST'
