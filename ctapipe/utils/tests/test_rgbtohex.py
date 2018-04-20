from ctapipe.utils.rgbtohex import intensity_to_rgb, intensity_to_hex
import numpy as np


def test_rgb():
    input_ = np.array([4])
    min_ = 0
    max_ = 10
    output = intensity_to_rgb(input_, min_, max_)

    assert (output == np.array([41, 120, 142, 255])).all()


def test_hex():
    input_ = np.array([4])
    min_ = 0
    max_ = 10
    output = intensity_to_hex(input_, min_, max_)

    assert (output == np.array(["#29788e"])).all()
