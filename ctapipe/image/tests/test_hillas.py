from ctapipe.instrument import CameraGeometry
from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.hillas import (hillas_parameters_1, hillas_parameters_2,
                                  hillas_parameters_3, hillas_parameters_4,
                                  HillasParameterizationError)
from ctapipe.io.containers import HillasParametersContainer
from astropy import units as u
from numpy import isclose, zeros_like, arange
from numpy.random import seed
from numpy.ma import masked_array
import pytest


def create_sample_image(psi='-30d'):

    seed(10)

    # set up the sample image using a HESS camera geometry (since it's easy
    # to load)
    geom = CameraGeometry.from_name("LSTCam")

    # make a toymodel shower model
    model = toymodel.generate_2d_shower_model(centroid=(0.2, 0.3),
                                              width=0.001, length=0.01,
                                              psi=psi)

    # generate toymodel image in camera for this shower model.
    image, signal, noise = toymodel.make_toymodel_shower_image(geom, model.pdf,
                                                               intensity=50,
                                                               nsb_level_pe=100)

    # denoise the image, so we can calculate hillas params
    clean_mask = tailcuts_clean(geom, image, 10,
                                5)  # pedvars = 1 and core and boundary

    return geom, image, clean_mask


def create_sample_image_zeros(psi='-30d'):

    geom, image, clean_mask = create_sample_image(psi)

    # threshold in pe
    image[~clean_mask] = 0

    return geom, image


def create_sample_image_masked(psi='-30d'):
    geom, image, clean_mask = create_sample_image(psi)

    # threshold in pe
    image = masked_array(image, mask=~clean_mask)

    return geom, image


def compare_result(x, y):
    ux = u.Quantity(x)
    uy = u.Quantity(y)
    assert isclose(ux.value, uy.value)
    assert ux.unit == uy.unit


def test_hillas():
    """
    test all Hillas-parameter routines on a sample image and see if they
    agree with eachother and with the toy model (assuming the toy model code
    is correct)
    """

    # try all quadrants
    for psi_angle in ['30d', '120d', '-30d', '-120d']:

        geom, image = create_sample_image_zeros(psi_angle)
        results = {}

        results['v1'] = hillas_parameters_1(geom, image)
        results['v2'] = hillas_parameters_2(geom, image)
        results['v3'] = hillas_parameters_3(geom, image)
        results['v4'] = hillas_parameters_4(geom, image)
        # compare each method's output
        for aa in results:
            for bb in results:
                if aa is not bb:
                    print("comparing {} to {}".format(aa, bb))
                    compare_result(results[aa].length, results[bb].length)
                    compare_result(results[aa].width, results[bb].width)
                    compare_result(results[aa].r, results[bb].r)
                    compare_result(results[aa].phi.deg, results[bb].phi.deg)
                    compare_result(results[aa].psi.deg, results[bb].psi.deg)
                    compare_result(results[aa].miss, results[bb].miss)
                    compare_result(results[aa].skewness, results[bb].skewness)
                    # compare_result(results[aa].kurtosis, results[bb].kurtosis)


def test_hillas_masked():
    """
    test Hillas-parameter routines on a sample image with masked values set to
    zero against a sample image with values masked with a numpy.ma.masked_array
    """

    geom, image = create_sample_image_zeros()
    geom, image_ma = create_sample_image_masked()

    results = hillas_parameters_4(geom, image)
    results_ma = hillas_parameters_4(geom, image_ma)

    compare_result(results.length, results_ma.length)
    compare_result(results.width, results_ma.width)
    compare_result(results.r, results_ma.r)
    compare_result(results.phi.deg, results_ma.phi.deg)
    compare_result(results.psi.deg, results_ma.psi.deg)
    compare_result(results.miss, results_ma.miss)
    compare_result(results.skewness, results_ma.skewness)
    # compare_result(results.kurtosis, results_ma.kurtosis)


def test_hillas_failure():
    geom, image = create_sample_image_zeros(psi='0d')
    blank_image = zeros_like(image)

    with pytest.raises(HillasParameterizationError):
        hillas_parameters_1(geom, blank_image)

    with pytest.raises(HillasParameterizationError):
        hillas_parameters_2(geom, blank_image)

    with pytest.raises(HillasParameterizationError):
        hillas_parameters_3(geom, blank_image)

    with pytest.raises(HillasParameterizationError):
        hillas_parameters_4(geom, blank_image)


def test_hillas_api_change():
    import numpy as np
    with pytest.raises(ValueError):
        hillas_parameters_4(arange(10), arange(10), arange(10))


def test_hillas_container():
    geom, image = create_sample_image_zeros(psi='0d')
    params = hillas_parameters_4(geom, image, container=True)
    assert type(params) is HillasParametersContainer
