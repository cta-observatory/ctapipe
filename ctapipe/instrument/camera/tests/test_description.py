from ctapipe.instrument import CameraDescription


def test_known_camera_names(camera_geometry):
    """Check that we can get a list of known camera names"""
    assert camera_geometry.name in CameraDescription.get_known_camera_names()
