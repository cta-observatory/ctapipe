""" Tests for TelescopeDescriptions """
import pytest

from ctapipe.instrument.telescope import TelescopeDescription


def test_hash(subarray_prod5_paranal):
    assert len(subarray_prod5_paranal) == 180
    assert len(set(subarray_prod5_paranal.tel.values())) == 4


@pytest.mark.parametrize("camera_name", ["LSTCam", "FlashCam", "NectarCam", "CHEC"])
@pytest.mark.parametrize("optics_name", ["LST", "MST", "ASTRI"])
def test_telescope_from_name(optics_name, camera_name, svc_path):
    """Check we can construct all telescopes from their names"""
    tel = TelescopeDescription.from_name(optics_name, camera_name)

    assert optics_name in str(tel)
    assert camera_name in str(tel)
    assert tel.camera.geometry.pix_x.shape[0] > 0
    assert tel.optics.equivalent_focal_length.to("m") > 0
