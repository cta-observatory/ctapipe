from importlib.resources import as_file

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable
from numpy.testing import assert_allclose

from ctapipe.coordinates import CameraFrame
from ctapipe.image.toymodel import Gaussian
from ctapipe.instrument import CameraGeometry, PixelShape
from ctapipe.utils.datasets import resource_file


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


@pytest.fixture
def veritas_cam_geom():
    with as_file(resource_file("veritas_pixel_coordinates.ecsv")) as path:
        table = QTable.read(path)

    pixel_area = 2 * np.sqrt(3) * table["r"] ** 2

    return CameraGeometry(
        name="VERITAS",
        pix_id=table["id"].copy(),
        pix_x=-table["y"].copy(),
        pix_y=-table["x"].copy(),
        pix_area=pixel_area,
        pix_rotation=30 * u.deg,
        pix_type=PixelShape.HEXAGON,
        frame=CameraFrame(),
    )


def test_single_image_veritas(veritas_cam_geom):
    """
    Regression test for #2778
    """
    image = create_mock_image(veritas_cam_geom)
    image_2d = veritas_cam_geom.image_to_cartesian_representation(image)
    image_1d = veritas_cam_geom.image_from_cartesian_representation(image_2d)
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
