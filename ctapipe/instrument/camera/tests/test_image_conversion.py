import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctapipe.image.toymodel import Gaussian


def create_mock_image(geom, psi=25 * u.deg):
    """
    creates a mock image, which parameters are adapted to the camera size
    """

    camera_r = np.max(np.sqrt(geom.pix_x**2 + geom.pix_y**2))
    model = Gaussian(
        x=0.3 * camera_r,
        y=0 * u.m,
        width=0.03 * camera_r,
        length=0.10 * camera_r,
        psi=psi,
    )

    _, image, _ = model.generate_image(
        geom, intensity=0.5 * geom.n_pixels, nsb_level_pe=3
    )
    return image


def test_single_image(camera_geometry):
    """
    Test if we can transform toy images for different geometries
    and get the same images after transforming back
    """
    image = create_mock_image(camera_geometry)
    image_2d = camera_geometry.image_to_cartesian_representation(image)
    image_1d = camera_geometry.image_from_cartesian_representation(image_2d)
    # in general this introduces extra pixels in the 2d array, which are set to nan
    assert np.nansum(image) == np.nansum(image_2d)
    assert_allclose(image, image_1d)


def test_multiple_images(camera_geometry):
    """
    Test if we can transform multiple toy images at once
    and get the same images after transforming back
    """
    images = np.array(
        [create_mock_image(camera_geometry, psi=i * 30 * u.deg) for i in range(5)]
    )
    images_2d = camera_geometry.image_to_cartesian_representation(images)
    images_1d = camera_geometry.image_from_cartesian_representation(images_2d)
    # in general this introduces extra pixels in the 2d array, which are set to nan
    assert np.nansum(images) == np.nansum(images_2d)
    assert_allclose(images, images_1d)


@pytest.mark.parametrize("pixel_id", [0, 1, 100])
def test_pixel_coordinates_roundtrip(pixel_id, camera_geometry):
    row, col = camera_geometry.image_index_to_cartesian_index(pixel_id)
    assert camera_geometry.cartesian_index_to_image_index(row, col) == pixel_id
