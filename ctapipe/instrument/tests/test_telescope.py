""" Tests for TelescopeDescriptions """
import pytest

from ctapipe.instrument.camera import CameraDescription
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
                TelescopeDescription(
                    name=name,
                    tel_type=type,
                    optics=OpticsDescription.from_name(name),
                    camera=CameraDescription.from_name(camera),
                )
            )

    assert len(telescopes) == 9
    assert len(set(telescopes)) == 3


@pytest.mark.parametrize("optics_name", ["LST", "MST"])
def test_telescope_from_name(optics_name, camera_geometry):
    """ Check we can construct all telescopes from their names """
    camera_name = camera_geometry.camera_name
    tel = TelescopeDescription.from_name(optics_name, camera_name)
    assert optics_name in str(tel)
    assert camera_name in str(tel)
    assert tel.camera.geometry.pix_x.shape[0] > 0
    assert tel.optics.equivalent_focal_length.to("m") > 0
    assert tel.type in {"MST", "SST", "LST", "UNKNOWN"}
