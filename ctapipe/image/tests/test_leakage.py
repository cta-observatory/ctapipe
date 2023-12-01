import numpy as np
import pytest

from ctapipe.containers import LeakageContainer
from ctapipe.instrument import CameraGeometry

# simple dummy test cases
geometry = CameraGeometry.make_rectangular(5, 5)
image_no_leakage = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
).ravel()
container_no_leakage = LeakageContainer(
    pixels_width_1=0.0,
    pixels_width_2=0.0,
    intensity_width_1=0.0,
    intensity_width_2=0.0,
)

image_leakage_2_1 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 8, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
).ravel()
container_leakage_2_1 = LeakageContainer(
    pixels_width_1=0.0,
    pixels_width_2=8 / 9,
    intensity_width_1=0.0,
    intensity_width_2=0.5,
)

image_leakage_2_2 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 9, 1, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ]
).ravel()
container_leakage_2_2 = LeakageContainer(
    pixels_width_1=0.0,
    pixels_width_2=2 / 3,
    intensity_width_1=0.0,
    intensity_width_2=0.25,
)

image_leakage_1 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 4, 1, 0, 0],
        [0, 1, 9, 2, 0],
        [0, 0, 2, 3, 1],
        [0, 0, 0, 1, 2],
    ]
).ravel()
container_leakage_1 = LeakageContainer(
    pixels_width_1=3 / 10,
    pixels_width_2=9 / 10,
    intensity_width_1=4 / 26,
    intensity_width_2=17 / 26,
)

images = (image_no_leakage, image_leakage_2_1, image_leakage_2_2, image_leakage_1)
containers = (
    container_no_leakage,
    container_leakage_2_1,
    container_leakage_2_2,
    container_leakage_1,
)


@pytest.mark.parametrize("image,expected", zip(images, containers))
def test_leakage_toy(image, expected):
    from ctapipe.image.leakage import leakage_parameters

    leakage = leakage_parameters(geometry, image, image > 0)

    for key, val in expected.items():
        assert leakage[key] == val, f"{key} does not match"


def test_leakage_lst(prod5_lst):
    from ctapipe.image.leakage import leakage_parameters

    geom = prod5_lst.camera.geometry

    img = np.ones(geom.n_pixels)
    mask = np.ones(len(geom), dtype=bool)

    leakage = leakage_parameters(geom, img, mask)

    ratio1 = np.sum(geom.get_border_pixel_mask(1)) / geom.n_pixels
    ratio2 = np.sum(geom.get_border_pixel_mask(2)) / geom.n_pixels

    assert leakage.intensity_width_1 == ratio1
    assert leakage.intensity_width_2 == ratio2
    assert leakage.pixels_width_1 == ratio1
    assert leakage.pixels_width_2 == ratio2
