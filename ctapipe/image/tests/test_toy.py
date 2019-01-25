import numpy as np
from ctapipe.instrument import CameraGeometry
from pytest import approx
from scipy.stats import poisson


def test_intensity():
    from .. import toymodel

    np.random.seed(0)

    geom = CameraGeometry.from_name('LSTCam')

    width = 0.05
    length = 0.15
    intensity = 50

    # make a toymodel shower model
    model = toymodel.generate_2d_shower_model(
        centroid=(0.2, 0.3),
        width=width, length=length,
        psi='30d',
    )

    image, signal, noise = toymodel.make_toymodel_shower_image(
        geom, model.pdf, intensity=intensity, nsb_level_pe=5,
    )

    # test if signal reproduces given cog values
    assert np.average(geom.pix_x.value, weights=signal) == approx(0.2, rel=0.15)
    assert np.average(geom.pix_y.value, weights=signal) == approx(0.3, rel=0.15)

    # test if signal reproduces given width/length values
    cov = np.cov(geom.pix_x.value, geom.pix_y.value, aweights=signal)
    eigvals, eigvecs = np.linalg.eigh(cov)

    assert np.sqrt(eigvals[0]) == approx(width, rel=0.15)
    assert np.sqrt(eigvals[1]) == approx(length, rel=0.15)

    # test if total intensity is inside in 99 percent confidence interval
    assert poisson(intensity).ppf(0.05) <= signal.sum() <= poisson(intensity).ppf(0.95)
