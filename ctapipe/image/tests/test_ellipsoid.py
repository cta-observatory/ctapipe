import itertools

import numpy as np
import numpy.ma as ma
import pytest
from astropy import units as u
from astropy.coordinates import Angle
from numpy import isclose
from numpy.testing import assert_allclose
from pytest import approx

from ctapipe.containers import (
    CameraImageFitParametersContainer,
    ImageFitParametersContainer,
)
from ctapipe.coordinates import TelescopeFrame
from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.concentration import concentration_parameters
from ctapipe.image.ellipsoid import ImageFitParameterizationError, image_fit_parameters
from ctapipe.instrument import CameraGeometry, SubarrayDescription


def create_sample_image(
    psi="-30d",
    x=0.05 * u.m,
    y=0.05 * u.m,
    width=0.1 * u.m,
    length=0.2 * u.m,
    intensity=1500,
    geometry=None,
):

    if geometry is None:
        s = SubarrayDescription.read("dataset://gamma_prod5.simtel.zst")
        geometry = s.tel[1].camera.geometry

    # make a toymodel shower model
    model = toymodel.Gaussian(x=x, y=y, width=width, length=length, psi=psi)

    # generate toymodel image in camera for this shower model.
    rng = np.random.default_rng(0)
    image, signal, noise = model.generate_image(
        geometry, intensity=intensity, nsb_level_pe=0, rng=rng
    )

    # calculate pixels likely containing signal
    clean_mask = tailcuts_clean(geometry, image, 10, 5)

    return image, clean_mask


def compare_result(x, y):
    if np.isnan(x) and np.isnan(y):
        x = 0
        y = 0
    ux = u.Quantity(x)
    uy = u.Quantity(y)
    assert isclose(ux.value, uy.value)
    assert ux.unit == uy.unit


def compare_fit(fit1, fit2):
    fit1_dict = fit1.as_dict()
    fit2_dict = fit2.as_dict()
    for key in fit1_dict.keys():
        compare_result(fit1_dict[key], fit2_dict[key])


def test_fit_selected(prod5_lst):
    """
    test Hillas-parameter routines on a sample image with selected values
    against a sample image with masked values set to zero
    """
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(geometry=geom)

    image_zeros = image.copy()
    image_zeros[~clean_mask] = 0.0

    image_selected = ma.masked_array(image.copy(), mask=~clean_mask)

    results = image_fit_parameters(
        geom, image_zeros, n=2, cleaned_mask=clean_mask, spe_width=0.5
    )
    results_selected = image_fit_parameters(
        geom, image_selected, n=2, cleaned_mask=clean_mask, spe_width=0.5
    )
    compare_fit(results, results_selected)


def test_dilation(prod5_lst):
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(geometry=geom)

    results = image_fit_parameters(
        geom,
        image,
        n=0,
        cleaned_mask=clean_mask,
        spe_width=0.5,
    )

    assert_allclose(results.intensity, np.sum(image[clean_mask]), rtol=1e-4, atol=1e-4)


def test_imagefit_failure(prod5_lst):
    geom = prod5_lst.camera.geometry
    blank_image = np.zeros(geom.n_pixels)

    with pytest.raises(ImageFitParameterizationError):
        image_fit_parameters(
            geom, blank_image, n=2, cleaned_mask=(blank_image == 1), spe_width=0.5
        )


def test_fit_container(prod5_lst):
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(psi="0d", geometry=geom)

    params = image_fit_parameters(
        geom, image, n=2, cleaned_mask=clean_mask, spe_width=0.5
    )
    assert isinstance(params, CameraImageFitParametersContainer)

    geom_telescope_frame = geom.transform_to(TelescopeFrame())
    params = image_fit_parameters(
        geom_telescope_frame, image, n=2, cleaned_mask=clean_mask, spe_width=0.5
    )
    assert isinstance(params, ImageFitParametersContainer)


def test_truncated(prod5_lst):
    rng = np.random.default_rng(42)
    geom = prod5_lst.camera.geometry

    width = 0.05 * u.m
    length = 0.30 * u.m
    intensity = 2000

    xs = u.Quantity([0.7, 0.8, -0.7, -0.6], u.m)
    ys = u.Quantity([0.6, -0.8, 0.6, -0.8], u.m)
    psis = Angle([-90, -45, 0, 45, 90], unit="deg")

    for x, y in zip(xs, ys):
        for psi in psis:

            # make a toymodel shower model
            model_gaussian = toymodel.Gaussian(
                x=x, y=y, width=width, length=length, psi=psi
            )

            image, signal, noise = model_gaussian.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )

            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom,
                signal,
                n=0,
                cleaned_mask=clean_mask,
                spe_width=0.5,
                pdf="Gaussian",
            )

            if result.is_valid & result.is_accurate:
                assert np.round(result.length, 1) >= length
                assert result.intensity == signal.sum()
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)


def test_percentage(prod5_lst):
    rng = np.random.default_rng(42)
    geom = prod5_lst.camera.geometry

    width = 0.03 * u.m
    length = 0.30 * u.m
    intensity = 2000

    xs = u.Quantity([0.2, 0.2, -0.2, -0.2], u.m)
    ys = u.Quantity([0.2, -0.2, 0.2, -0.2], u.m)
    psis = Angle([-90, -45, 0, 45, 90], unit="deg")

    for x, y in zip(xs, ys):
        for psi in psis:
            # make a toymodel shower model
            model_gaussian = toymodel.Gaussian(
                x=x, y=y, width=width, length=length, psi=psi
            )

            image, signal, noise = model_gaussian.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )
            clean_mask = signal > 0
            fit = image_fit_parameters(
                geom,
                signal,
                n=0,
                cleaned_mask=clean_mask,
                spe_width=0.5,
                pdf="Gaussian",
            )

            conc = concentration_parameters(geom, signal, fit)
            signal_inside_ellipse = conc.core

            if fit.is_valid and fit.is_accurate:
                assert signal_inside_ellipse > 0.5


def test_with_toy(prod5_lst):
    rng = np.random.default_rng(42)

    geom = prod5_lst.camera.geometry

    width = 0.03 * u.m
    length = 0.15 * u.m
    width_uncertainty = 0.00094 * u.m
    length_uncertainty = 0.00465 * u.m
    intensity = 500

    xs = u.Quantity([0.2, 0.2, -0.2, -0.2], u.m)
    ys = u.Quantity([0.2, -0.2, 0.2, -0.2], u.m)
    psis = Angle([-90, -45, 0, 45, 90], unit="deg")

    for x, y in zip(xs, ys):
        for psi in psis:

            # make a toymodel shower model
            model_gaussian = toymodel.Gaussian(
                x=x, y=y, width=width, length=length, psi=psi
            )
            model_skewed = toymodel.SkewedGaussian(
                x=x, y=y, width=width, length=length, psi=psi, skewness=0.5
            )
            model_cauchy = toymodel.SkewedCauchy(
                x=x, y=y, width=width, length=length, psi=psi, skewness=0.5
            )

            image, signal, noise = model_gaussian.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )

            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom,
                signal,
                n=0,
                cleaned_mask=clean_mask,
                spe_width=0.5,
                pdf="Gaussian",
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.3)
                assert u.isclose(result.y, y, rtol=0.3)

                assert u.isclose(result.width, width, rtol=1)
                assert u.isclose(result.width_uncertainty, width_uncertainty, rtol=1)
                assert u.isclose(result.length, length, rtol=1)
                assert u.isclose(result.length_uncertainty, length_uncertainty, rtol=1)
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)

            image, signal, noise = model_skewed.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )

            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom, signal, n=0, cleaned_mask=clean_mask, spe_width=0.5, pdf="Skewed"
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.3)
                assert u.isclose(result.y, y, rtol=0.3)

                assert u.isclose(result.width, width, rtol=1)
                assert u.isclose(result.width_uncertainty, width_uncertainty, rtol=1)
                assert u.isclose(result.length, length, rtol=1)
                assert u.isclose(result.length_uncertainty, length_uncertainty, rtol=1)
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)

            image, signal, noise = model_cauchy.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )

            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom, signal, n=0, cleaned_mask=clean_mask, spe_width=0.5
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.3)
                assert u.isclose(result.y, y, rtol=0.3)

                assert u.isclose(result.width, width, rtol=1)
                assert u.isclose(result.width_uncertainty, width_uncertainty, rtol=1)
                assert u.isclose(result.length, length, rtol=1)
                assert u.isclose(result.length_uncertainty, length_uncertainty, rtol=1)
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)


def test_skewness(prod5_lst):
    rng = np.random.default_rng(42)
    geom = prod5_lst.camera.geometry

    intensity = 2500

    widths = u.Quantity([0.04, 0.06], u.m)
    lengths = u.Quantity([0.20, 0.30], u.m)
    xs = u.Quantity([0.3, 0.3, -0.3, -0.3, 0, 0.1], u.m)
    ys = u.Quantity([0.3, -0.3, 0.3, -0.3, 0, 0.2], u.m)
    psis = Angle([-90, -45, 0, 45, 90], unit="deg")
    skews = [0, 0.3, 0.6, 0.8]

    for x, y, psi, skew, width, length in itertools.product(
        xs, ys, psis, skews, widths, lengths
    ):
        # make a toymodel shower model
        model = toymodel.SkewedGaussian(
            x=x, y=y, width=width, length=length, psi=psi, skewness=skew
        )

        _, signal, _ = model.generate_image(
            geom, intensity=intensity, nsb_level_pe=0, rng=rng
        )
        clean_mask = np.array(signal) > 0

        result = image_fit_parameters(
            geom, signal, n=0, cleaned_mask=clean_mask, spe_width=0.5, pdf="Skewed"
        )

        if (result.is_valid == True) or (result.is_accurate == True):
            assert u.isclose(result.x, x, rtol=0.5)
            assert u.isclose(result.y, y, rtol=0.5)

            assert u.isclose(result.width, width, rtol=0.5)
            assert u.isclose(result.length, length, rtol=0.5)

            psi_same = result.psi.to_value(u.deg) == approx(psi.deg, abs=3)
            psi_opposite = abs(result.psi.to_value(u.deg) - psi.deg) == approx(
                180.0, abs=3
            )

            assert psi_same or psi_opposite

            if psi_same:
                assert result.skewness == approx(skew, abs=0.3)
            else:
                assert result.skewness == approx(-skew, abs=0.3)

            assert signal.sum() == result.intensity


@pytest.mark.filterwarnings("error")
def test_single_pixel():
    x = y = np.arange(3)
    x, y = np.meshgrid(x, y)

    geom = CameraGeometry(
        name="testcam",
        pix_id=np.arange(9),
        pix_x=x.ravel() * u.cm,
        pix_y=y.ravel() * u.cm,
        pix_type="rectangular",
        pix_area=1 * u.cm**2,
    )

    image = np.zeros((3, 3))
    image[1, 1] = 10
    image = image.ravel()

    clean_mask = np.array(image) > 0

    with pytest.raises(ImageFitParameterizationError):
        image_fit_parameters(geom, image, n=2, cleaned_mask=clean_mask, spe_width=0.5)

    with pytest.raises(ImageFitParameterizationError):
        image_fit_parameters(
            geom, image, n=2, cleaned_mask=clean_mask, spe_width=0.5, pdf="Skewed"
        )

    with pytest.raises(ImageFitParameterizationError):
        image_fit_parameters(
            geom, image, n=2, cleaned_mask=clean_mask, spe_width=0.5, pdf="Gaussian"
        )
