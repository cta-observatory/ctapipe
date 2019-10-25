from ctapipe.instrument.camera import CameraGeometry
import numpy as np


def test_leakage():
    from ctapipe.image.leakage import leakage

    geom = CameraGeometry.from_name('LSTCam')

    img = np.ones(geom.n_pixels)
    mask = np.ones(len(geom), dtype=bool)

    l = leakage(geom, img, mask)

    ratio1 = np.sum(geom.get_border_pixel_mask(1)) / geom.n_pixels
    ratio2 = np.sum(geom.get_border_pixel_mask(2)) / geom.n_pixels

    assert l.intensity_width_1 == ratio1
    assert l.intensity_width_2 == ratio2
    assert l.pixels_width_1 == ratio1
    assert l.pixels_width_2 == ratio2
