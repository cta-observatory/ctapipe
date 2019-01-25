from ctapipe.image.tests.test_hillas import create_sample_image
from ctapipe.image.hillas import hillas_parameters
from ctapipe.image.concentration import concentration


def test_concentration():
    geom, image, clean_mask = create_sample_image('30d')

    hillas = hillas_parameters(geom[clean_mask], image[clean_mask])

    conc = concentration(geom, image, hillas)

    assert 0.1 <= conc.concentration_cog <= 0.3
    assert 0.05 <= conc.concentration_pixel <= 0.2
    assert 0.3 <= conc.concentration_core <= 0.6


if __name__ == '__main__':
    test_concentration()
