"""Tests for the data structures defined in ctapipe.containers"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from ctapipe.containers import (
    DL0CameraContainer,
    ImageParametersContainer,
    SubarrayDL2Container,
    SubarrayEventContainer,
    TelescopeDL0Container,
    TelescopeDL1Container,
    TelescopeEventContainer,
)


@pytest.mark.parametrize(
    "container_type", [SubarrayEventContainer, TelescopeEventContainer]
)
def test_absent_data_levels(container_type):
    event = container_type()
    levels = ("simulation", "dl0", "dl2")
    if container_type is TelescopeEventContainer:
        levels += ("r0", "r1", "dl1", "muon")
    for level in levels:
        assert getattr(event, level) is None
    event.validate()


def test_present_data_level_with_missing_information():
    event = TelescopeEventContainer(dl1=TelescopeDL1Container())
    assert event.dl1 is not None
    assert event.dl1.image is None
    assert event.dl1.parameters is None
    event.validate()
    event.reset()
    assert event.dl1 is None


def test_dl0_trigger_without_camera():
    dl0 = TelescopeDL0Container()
    assert dl0.trigger is not None
    assert dl0.camera is None
    dl0.camera = DL0CameraContainer()
    assert dl0.camera.waveform is None
    dl0.validate()


@pytest.mark.parametrize(
    "product", ["dl1", "dl1.image", "dl1.parameters", "dl0.camera"]
)
def test_inconsistent_telescope_data_levels(product):
    event = SubarrayEventContainer()
    for tel_id in (1, 2):
        event.tel[tel_id].dl1 = TelescopeDL1Container()
        event.tel[tel_id].dl0 = TelescopeDL0Container()
    if product == "dl1":
        event.tel[2].dl1 = None
    elif product == "dl1.image":
        event.tel[1].dl1.image = np.ones(3)
    elif product == "dl1.parameters":
        event.tel[1].dl1.parameters = ImageParametersContainer()
    else:
        event.tel[1].dl0.camera = DL0CameraContainer()
    with pytest.raises(ValueError, match=product):
        event.validate_data_levels()


def test_subarray_dl2_independent_of_telescope_dl2():
    event = SubarrayEventContainer(dl2=SubarrayDL2Container())
    event.tel[1] = TelescopeEventContainer()
    event.tel[2] = TelescopeEventContainer()
    event.validate_data_levels()


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
