from ctapipe.instrument import CameraGeometry
from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.io.containers import HillasParametersContainer
from astropy.coordinates import Angle
from astropy import units as u
import numpy as np
from numpy import isclose, zeros_like
from numpy.random import seed
from pytest import approx
import pytest


def create_sample_image(
        psi='-30d',
        centroid=(0.2, 0.3),
        width=0.05,
        length=0.15,
        intensity=1500
):
    seed(10)

    geom = CameraGeometry.from_name('LSTCam')

    # make a toymodel shower model
    model = toymodel.generate_2d_shower_model(
        centroid=centroid,
        width=width,
        length=length,
        psi=psi,
    )

    # generate toymodel image in camera for this shower model.
    image, signal, noise = toymodel.make_toymodel_shower_image(
        geom, model.pdf,
        intensity=1500,
        nsb_level_pe=3,
    )

    # denoise the image, so we can calculate hillas params
    clean_mask = tailcuts_clean(geom, image, 10, 5)

    return geom, image, clean_mask


def create_sample_image_zeros(psi='-30d'):

    geom, image, clean_mask = create_sample_image(psi)

    # threshold in pe
    image[~clean_mask] = 0

    return geom, image


def create_sample_image_selected_pixel(psi='-30d'):
    geom, image, clean_mask = create_sample_image(psi)

    return geom[clean_mask], image[clean_mask]


def compare_result(x, y):
    ux = u.Quantity(x)
    uy = u.Quantity(y)
    assert isclose(ux.value, uy.value)
    assert ux.unit == uy.unit


def test_hillas_selected():
    """
    test Hillas-parameter routines on a sample image with selected values
    against a sample image with masked values set tozero
    """
    geom, image = create_sample_image_zeros()
    geom_selected, image_selected = create_sample_image_selected_pixel()

    results = hillas_parameters(geom, image)
    results_selected = hillas_parameters(geom_selected, image_selected)

    compare_result(results.length, results_selected.length)
    compare_result(results.width, results_selected.width)
    compare_result(results.r, results_selected.r)
    compare_result(results.phi.deg, results_selected.phi.deg)
    compare_result(results.psi.deg, results_selected.psi.deg)
    compare_result(results.skewness, results_selected.skewness)
    # compare_result(results.kurtosis, results_ma.kurtosis)


def test_hillas_failure():
    geom, image = create_sample_image_zeros(psi='0d')
    blank_image = zeros_like(image)

    with pytest.raises(HillasParameterizationError):
        hillas_parameters(geom, blank_image)


def test_hillas_masked_array():
    geom_zeros, image_zeros = create_sample_image_zeros()
    hillas_zeros = hillas_parameters(geom_zeros, image_zeros)

    geom_masked, image, clean_mask = create_sample_image(psi='0d')
    image_masked = np.ma.masked_array(image, mask=~clean_mask)
    hillas_masked = hillas_parameters(geom_masked, image_masked)

    compare_result(hillas_zeros.length, hillas_masked.length)
    compare_result(hillas_zeros.width, hillas_masked.width)
    compare_result(hillas_zeros.r, hillas_masked.r)
    compare_result(hillas_zeros.phi.deg, hillas_masked.phi.deg)
    compare_result(hillas_zeros.psi.deg, hillas_masked.psi.deg)
    compare_result(hillas_zeros.skewness, hillas_masked.skewness)
    compare_result(hillas_zeros.kurtosis, hillas_masked.kurtosis)


def test_hillas_container():
    geom, image = create_sample_image_zeros(psi='0d')

    params = hillas_parameters(geom, image)
    assert isinstance(params, HillasParametersContainer)


def test_with_toy():
    np.random.seed(42)

    geom = CameraGeometry.from_name('LSTCam')

    width = 0.03
    length = 0.15
    intensity = 500

    xs = (0.5, 0.5, -0.5, -0.5)
    ys = (0.5, -0.5, 0.5, -0.5)
    psis = Angle([-90, -45, 0, 45, 90], unit='deg')

    for x, y in zip(xs, ys):
        for psi in psis:

            # make a toymodel shower model
            model = toymodel.generate_2d_shower_model(
                centroid=(x, y),
                width=width, length=length,
                psi=psi,
            )

            image, signal, noise = toymodel.make_toymodel_shower_image(
                geom, model.pdf, intensity=intensity, nsb_level_pe=5,
            )

            result = hillas_parameters(geom, signal)

            assert result.x.to_value(u.m) == approx(x, rel=0.1)
            assert result.y.to_value(u.m) == approx(y, rel=0.1)

            assert result.width.to_value(u.m) == approx(width, rel=0.1)
            assert result.length.to_value(u.m) == approx(length, rel=0.1)
            assert (
                (result.psi.to_value(u.deg) == approx(psi.deg, abs=2))
                or abs(result.psi.to_value(u.deg) - psi.deg) == approx(180.0, abs=2)
            )

            assert signal.sum() == result.intensity
