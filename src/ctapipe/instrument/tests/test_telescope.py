""" Tests for TelescopeDescriptions """
import pytest

from ctapipe.instrument import FromNameWarning
from ctapipe.instrument.telescope import TelescopeDescription


def test_repr(subarray_prod5_paranal):
    expected = (
        "TelescopeDescription(type='LST', optics_name='LST', camera_name='LSTCam')"
    )
    assert repr(subarray_prod5_paranal.tel[1]) == expected

    expected = (
        "TelescopeDescription(type='MST', optics_name='MST', camera_name='FlashCam')"
    )
    assert repr(subarray_prod5_paranal.tel[5]) == expected

    expected = (
        "TelescopeDescription(type='SST', optics_name='ASTRI', camera_name='CHEC')"
    )
    assert repr(subarray_prod5_paranal.tel[50]) == expected


def test_str(subarray_prod5_paranal):
    assert str(subarray_prod5_paranal.tel[1]) == "LST_LST_LSTCam"
    assert str(subarray_prod5_paranal.tel[5]) == "MST_MST_FlashCam"
    assert str(subarray_prod5_paranal.tel[50]) == "SST_ASTRI_CHEC"


def test_hash(subarray_prod5_paranal):
    assert len(subarray_prod5_paranal) == 180
    assert len(set(subarray_prod5_paranal.tel.values())) == 4


@pytest.mark.parametrize("camera_name", ["LSTCam", "FlashCam", "NectarCam", "CHEC"])
@pytest.mark.parametrize("optics_name", ["LST", "MST", "ASTRI"])
def test_telescope_from_name(optics_name, camera_name, svc_path):
    """Check we can construct all telescopes from their names"""
    with pytest.warns(FromNameWarning):
        tel = TelescopeDescription.from_name(optics_name, camera_name)

    assert optics_name in str(tel)
    assert camera_name in str(tel)
    assert tel.camera.geometry.pix_x.shape[0] > 0
    assert tel.optics.equivalent_focal_length.to("m") > 0
