import pytest
import numpy as np
from numpy.testing import assert_allclose
from ctapipe.image.geometry_converter import (
    convert_geometry_rect2d_back_to_hex1d,
    convert_geometry_hex1d_to_rect2d,
    convert_rect_image_1d_to_2d,
    convert_rect_image_back_to_1d,
)
from ctapipe.instrument import CameraGeometry, PixelShape
from ctapipe.image.toymodel import Gaussian
import astropy.units as u


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
def test_convert_geometry(camera_geometry, rot):
    """
    Test if we can transform toy images for different geometries
    and get the same images after transforming back
    """
    image = create_mock_image(camera_geometry)

    if camera_geometry.pix_type is PixelShape.HEXAGON:
        geom_2d, image_2d = convert_geometry_hex1d_to_rect2d(
            camera_geometry, image, camera_geometry.camera_name + str(rot), add_rot=rot
        )
        geom_1d, image_1d = convert_geometry_rect2d_back_to_hex1d(
            geom_2d, image_2d, camera_geometry.camera_name + str(rot), add_rot=rot
        )

    else:
        rows_cols, image_2d = convert_rect_image_1d_to_2d(camera_geometry, image)
        image_1d = convert_rect_image_back_to_1d(rows_cols, image_2d)
    assert_allclose(image, image_1d)


@pytest.mark.parametrize("rot", [3])
def test_convert_geometry_mock(camera_geometry, rot):
    """here we use a different key for the back conversion to trigger the mock conversion
    """

    image = create_mock_image(camera_geometry)

    if camera_geometry.pix_type is PixelShape.HEXAGON:
        convert_geometry_1d_to_2d = convert_geometry_hex1d_to_rect2d
        convert_geometry_back = convert_geometry_rect2d_back_to_hex1d

        geom2d, image2d = convert_geometry_1d_to_2d(
            camera_geometry, image, key=None, add_rot=rot
        )
        geom1d, image1d = convert_geometry_back(
            geom2d,
            image2d,
            "_".join([camera_geometry.camera_name, str(rot), "mock"]),
            add_rot=rot,
        )
    else:
        # originally rectangular geometries don't need a buffer and therefore no mock
        # conversion
        return
