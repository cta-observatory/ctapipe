from ctapipe.instrument import CameraDescription


def test_known_camera_names(camera_geometries):
    """ Check that we can get a list of known camera names """
    cams = CameraDescription.get_known_camera_names()
    assert len(cams) > 4
    assert "FlashCam" in cams
    assert "NectarCam" in cams

    # TODO: Requires camreadout files to be generated
    # for cam in cams:
    #     camera = CameraDescription.from_name(cam)
    #     camera.info()
