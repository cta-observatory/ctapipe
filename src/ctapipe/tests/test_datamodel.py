"""Tests for the data structures defined in ctapipe.containers"""
import numpy as np
from numpy.testing import assert_equal


def test_pixel_status():
    """Test methods of the PixelStatus enum on numpy arrays"""
    from ctapipe.containers import PixelStatus

    pixel_status = np.array([0b1101, 0b1000, 0b1110, 0b1101, 0b0000], dtype=np.uint8)

    assert_equal(
        PixelStatus.is_invalid(pixel_status), [False, False, False, False, True]
    )
    assert_equal(PixelStatus.get_dvr_status(pixel_status), [1, 0, 2, 1, 0])
    assert_equal(
        PixelStatus.get_channel_info(pixel_status), [0b11, 0b10, 0b11, 0b11, 0b00]
    )
