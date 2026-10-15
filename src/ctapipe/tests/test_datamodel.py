"""Tests for the data structures defined in ctapipe.containers"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from ctapipe.containers import PixelStatus


def test_pixel_status():
    """Test methods of the PixelStatus enum on numpy arrays"""
    pixel_status = np.array([0b1101, 0b1000, 0b1110, 0b1101, 0b0000], dtype=np.uint8)

    assert_equal(
        PixelStatus.is_invalid(pixel_status), [False, False, False, False, True]
    )
    assert_equal(PixelStatus.get_dvr_status(pixel_status), [1, 0, 2, 1, 0])
    assert_equal(
        PixelStatus.get_channel_info(pixel_status), [0b11, 0b10, 0b11, 0b11, 0b00]
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (PixelStatus.HIGH_GAIN_STORED, 0),
        (PixelStatus.LOW_GAIN_STORED, 1),
        (PixelStatus.HIGH_GAIN_STORED | PixelStatus.LOW_GAIN_STORED, 0),
        (PixelStatus(0), 0),
    ],
)
def test_pixel_status_selected_channel(status, expected):
    assert PixelStatus.selected_channel(status) == expected

    # Other status bits must not affect the selected gain.
    status = np.uint8(status)
    statuses = np.array(
        [status, status | PixelStatus.DVR_0, status | PixelStatus.SATURATED],
        dtype=np.uint8,
    )
    channels = PixelStatus.selected_channel(statuses)
    assert channels.dtype == np.uint8
    assert_equal(channels, [expected, expected, expected])
