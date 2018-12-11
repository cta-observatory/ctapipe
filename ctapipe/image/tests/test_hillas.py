from ctapipe.instrument import CameraGeometry
from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.io.containers import HillasParametersContainer
from astropy import units as u
from numpy import isclose, zeros_like, arange
from numpy.random import seed
import pytest


def create_sample_image(psi='-30d', centroid=(0.2, 0.3), width=0.05, length=0.15, intensity=1500):
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


def test_hillas_api_change():
    with pytest.raises(TypeError):
        hillas_parameters(arange(10), arange(10), arange(10))


def test_hillas_container():
    geom, image = create_sample_image_zeros(psi='0d')

    params = hillas_parameters(geom, image)
    assert isinstance(params, HillasParametersContainer)
