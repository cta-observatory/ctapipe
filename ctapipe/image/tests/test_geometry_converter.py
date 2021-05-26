import pytest
import numpy as np
from ctapipe.image.geometry_converter import (
    convert_geometry_rect2d_back_to_hexe1d,
    convert_geometry_hex1d_to_rect2d,
    convert_rect_image_1d_to_2d,
    convert_rect_image_back_to_1d,
)
from ctapipe.image.hillas import hillas_parameters
from ctapipe.instrument import CameraDescription, CameraGeometry, PixelShape
from ctapipe.image.toymodel import Gaussian
import astropy.units as u
from numpy.testing import assert_allclose


camera_names = CameraDescription.get_known_camera_names()


def create_mock_image(geom):
    """
    creates a mock image, which parameters are adapted to the camera size
    """

    camera_r = np.max(np.sqrt(geom.pix_x ** 2 + geom.pix_y ** 2))
    model = Gaussian(
        x=0.3 * camera_r,
        y=0 * u.m,
        width=0.03 * camera_r,
        length=0.10 * camera_r,
        psi="25d",
    )

    _, image, _ = model.generate_image(
        geom, intensity=0.5 * geom.n_pixels, nsb_level_pe=3
    )
    return image


@pytest.mark.parametrize("rot", [3])
@pytest.mark.parametrize("camera_name", camera_names)
def test_convert_geometry(camera_name, rot):

    geom = CameraGeometry.from_name(camera_name)
    image = create_mock_image(geom)
    hillas_0 = hillas_parameters(geom, image)

    if geom.pix_type is PixelShape.HEXAGON:
        geom2d, image2d = convert_geometry_hex1d_to_rect2d(
            geom, image, geom.camera_name + str(rot), add_rot=rot
        )
        geom1d, image1d = convert_geometry_rect2d_back_to_hexe1d(
            geom2d, image2d, geom.camera_name + str(rot), add_rot=rot
        )

    else:
        image2d, r, c = convert_rect_image_1d_to_2d(image, geom)
        image1d = convert_rect_image_back_to_1d(image2d, r, c)
    assert_allclose(image, image1d)


@pytest.mark.parametrize("rot", [3])
@pytest.mark.parametrize("camera_name", camera_names)
def test_convert_geometry_mock(camera_name, rot):
    """here we use a different key for the back conversion to trigger the mock conversion
    """

    geom = CameraGeometry.from_name(camera_name)
    image = create_mock_image(geom)
    hillas_0 = hillas_parameters(geom, image)

    if geom.pix_type == "hexagonal":
        convert_geometry_1d_to_2d = convert_geometry_hex1d_to_rect2d
        convert_geometry_back = convert_geometry_rect2d_back_to_hexe1d

        geom2d, image2d = convert_geometry_1d_to_2d(geom, image, key=None, add_rot=rot)
        geom1d, image1d = convert_geometry_back(
            geom2d, image2d, "_".join([geom.camera_name, str(rot), "mock"]), add_rot=rot
        )
    else:
        # originally rectangular geometries don't need a buffer and therefore no mock
        # conversion
        return

    assert_allclose(image, image1d)
