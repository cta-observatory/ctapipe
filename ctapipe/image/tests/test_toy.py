import numpy as np
from ctapipe.instrument import CameraGeometry
from pytest import approx
from scipy.stats import poisson, skewnorm
import astropy.units as u


def test_intensity():
    from ctapipe.image.toymodel import Gaussian

    np.random.seed(0)
    geom = CameraGeometry.from_name('LSTCam')

    x, y = u.Quantity([0.2, 0.3], u.m)
    width = 0.05 * u.m
    length = 0.15 * u.m
    intensity = 50
    psi = '30d'

    # make a toymodel shower model
    model = Gaussian(x=x, y=y, width=width, length=length, psi=psi)

    image, signal, noise = model.generate_image(
        geom, intensity=intensity, nsb_level_pe=5,
    )

    # test if signal reproduces given cog values
    assert np.average(geom.pix_x.to_value(u.m), weights=signal) == approx(0.2, rel=0.15)
    assert np.average(geom.pix_y.to_value(u.m), weights=signal) == approx(0.3, rel=0.15)

    # test if signal reproduces given width/length values
    cov = np.cov(geom.pix_x.value, geom.pix_y.value, aweights=signal)
    eigvals, eigvecs = np.linalg.eigh(cov)

    assert np.sqrt(eigvals[0]) == approx(width.to_value(u.m), rel=0.15)
    assert np.sqrt(eigvals[1]) == approx(length.to_value(u.m), rel=0.15)

    # test if total intensity is inside in 99 percent confidence interval
    assert poisson(intensity).ppf(0.05) <= signal.sum() <= poisson(intensity).ppf(0.95)


def test_skewed():
    from ctapipe.image.toymodel import SkewedGaussian
    # test if the parameters we calculated for the skew normal
    # distribution produce the correct moments
    np.random.seed(0)
    geom = CameraGeometry.from_name('LSTCam')

    x, y = u.Quantity([0.2, 0.3], u.m)
    width = 0.05 * u.m
    length = 0.15 * u.m
    intensity = 50
    psi = '30d'
    skewness = 0.3

    model = SkewedGaussian(
        x=x, y=y, width=width,
        length=length, psi=psi, skewness=skewness
    )
    image, signal, _ = model.generate_image(
        geom, intensity=intensity, nsb_level_pe=5,
    )

    a, loc, scale = model._moments_to_parameters()
    mean, var, skew = skewnorm(a=a, loc=loc, scale=scale).stats(moments='mvs')

    assert np.isclose(mean, 0)
    assert np.isclose(var, length.to_value(u.m)**2)
    assert np.isclose(skew, skewness)


def test_compare():
    from ctapipe.image.toymodel import SkewedGaussian, Gaussian
    geom = CameraGeometry.from_name('LSTCam')

    x, y = u.Quantity([0.2, 0.3], u.m)
    width = 0.05 * u.m
    length = 0.15 * u.m
    intensity = 50
    psi = '30d'

    skewed = SkewedGaussian(
        x=x, y=y, width=width,
        length=length, psi=psi, skewness=0
    )
    normal = Gaussian(
        x=x, y=y, width=width,
        length=length, psi=psi
    )

    signal_skewed = skewed.expected_signal(geom, intensity=intensity)
    signal_normal = normal.expected_signal(geom, intensity=intensity)

    assert np.isclose(signal_skewed, signal_normal).all()
