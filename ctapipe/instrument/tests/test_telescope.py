""" Tests for TelescopeDescriptions """
import pytest

from ctapipe.instrument.camera import CameraGeometry
from ctapipe.instrument.optics import OpticsDescription
from ctapipe.instrument.telescope import TelescopeDescription


def test_hash():

    types = ["LST", "MST", "SST"]
    names = ["LST", "MST", "SST-1M"]
    cameras = ["LSTCam", "FlashCam", "DigiCam"]

    telescopes = []
    for name, type, camera in zip(names, types, cameras):
        for i in range(3):

            telescopes.append(
                TelescopeDescription(name=name, tel_type=type,
                                     optics=OpticsDescription.from_name(name),
                                     camera=CameraGeometry.from_name(camera))
            )

    assert len(telescopes) == 9
    assert len(set(telescopes)) == 3


OPTICS_NAMES = OpticsDescription.get_known_optics_names()
CAMERA_NAMES = CameraGeometry.get_known_camera_names()


@pytest.mark.parametrize("camera_name", CAMERA_NAMES)
@pytest.mark.parametrize("optics_name", OPTICS_NAMES)
def test_telescope_from_name(optics_name, camera_name):
    """ Check we can construct all telescopes from their names """
    tel = TelescopeDescription.from_name(optics_name, camera_name)
    assert optics_name in str(tel)
    assert camera_name in str(tel)
    assert tel.camera.pix_x.shape[0] > 0
    assert tel.optics.equivalent_focal_length.to("m") > 0
    assert tel.type in ["MST", "SST", "LST", "UNKNOWN"]
