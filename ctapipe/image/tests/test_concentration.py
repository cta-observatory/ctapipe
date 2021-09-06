from ctapipe.image.tests.test_hillas import create_sample_image
from ctapipe.image.hillas import hillas_parameters
from ctapipe.image.concentration import concentration_parameters
import astropy.units as u
import pytest


def test_concentration():
    geom, image, clean_mask = create_sample_image("30d")

    hillas = hillas_parameters(geom[clean_mask], image[clean_mask])

    conc = concentration_parameters(geom[clean_mask], image[clean_mask], hillas)

    assert 0.1 <= conc.cog <= 0.3
    assert 0.04 <= conc.pixel <= 0.2
    assert 0.3 <= conc.core <= 0.6


@pytest.mark.filterwarnings("error")
def test_width_0():
    geom, image, clean_mask = create_sample_image("30d")

    hillas = hillas_parameters(geom[clean_mask], image[clean_mask])
    hillas.width = 0 * u.m

    conc = concentration_parameters(geom, image, hillas)
    assert conc.core == 0


def test_no_pixels_near_cog():
    geom, image, clean_mask = create_sample_image("30d")

    hillas = hillas_parameters(geom[clean_mask], image[clean_mask])

    # remove pixels close to cog from the cleaning pixels
    clean_mask &= ((geom.pix_x - hillas.x) ** 2 + (geom.pix_y - hillas.y) ** 2) >= (
        2 * geom.pixel_width ** 2
    )

    conc = concentration_parameters(geom[clean_mask], image[clean_mask], hillas)
    assert conc.cog == 0


if __name__ == "__main__":
    test_concentration()
