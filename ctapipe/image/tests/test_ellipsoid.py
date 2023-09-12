import numpy as np
import numpy.ma as ma
import pytest
from astropy import units as u
from astropy.coordinates import Angle
from numpy import isclose
from numpy.testing import assert_allclose
from pytest import approx

from ctapipe.containers import ImageFitParametersContainer
from ctapipe.image import hillas_parameters, tailcuts_clean, toymodel
from ctapipe.image.concentration import concentration_parameters
from ctapipe.image.ellipsoid import (
    ImageFitParameterizationError,
    PDFType,
    boundaries,
    image_fit_parameters,
)
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
        geometry, intensity=intensity, nsb_level_pe=3, rng=rng
    )

    # calculate pixels likely containing signal
    clean_mask = tailcuts_clean(geometry, image, 10, 5)

    return image, clean_mask


def test_boundaries(prod5_lst):
    # Test default functin for finding the boundaries of the fit
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(geometry=geom)

    cleaned_image = image.copy()
    cleaned_image[~clean_mask] = 0.0

    hillas = hillas_parameters(geom, cleaned_image)
    bounds = boundaries(geom, image, clean_mask, hillas, pdf=PDFType("gaussian"))

    for i in range(len(bounds)):
        assert bounds[i][1] > bounds[i][0]  # upper limit > lower limit


def compare_result(x, y):
    if np.isnan(x) and np.isnan(y):
        x = 0
        y = 0
    ux = u.Quantity(x)
    uy = u.Quantity(y)
    assert isclose(ux.value, uy.value)
    assert ux.unit == uy.unit


def compare_fit_params(fit1, fit2):
    fit1_dict = fit1.as_dict()
    fit2_dict = fit2.as_dict()
    for key in fit1_dict.keys():
        compare_result(fit1_dict[key], fit2_dict[key])


def test_fit_selected(prod5_lst):
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(geometry=geom)

    image_zeros = image.copy()
    image_zeros[~clean_mask] = 0.0

    image_selected = ma.masked_array(image.copy(), mask=~clean_mask)

    results = image_fit_parameters(
        geom,
        image_zeros,
        n_row=2,
        cleaned_mask=clean_mask,
    )
    results_selected = image_fit_parameters(
        geom,
        image_selected,
        n_row=2,
        cleaned_mask=clean_mask,
    )
    compare_fit_params(results, results_selected)


def test_dilation(prod5_lst):
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(geometry=geom)

    results = image_fit_parameters(
        geom,
        image,
        n_row=0,
        cleaned_mask=clean_mask,
    )

    assert_allclose(results.intensity, np.sum(image[clean_mask]), rtol=1e-4, atol=1e-4)

    results = image_fit_parameters(
        geom,
        image,
        n_row=2,
        cleaned_mask=clean_mask,
    )

    assert results.intensity > np.sum(image[clean_mask])


def test_imagefit_failure(prod5_lst):
    geom = prod5_lst.camera.geometry
    blank_image = np.zeros(geom.n_pixels)

    with pytest.raises(ImageFitParameterizationError):
        image_fit_parameters(
            geom,
            blank_image,
            n_row=2,
            cleaned_mask=(blank_image == 1),
        )


def test_fit_container(prod5_lst):
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(psi="0d", geometry=geom)

    params = image_fit_parameters(
        geom,
        image,
        n_row=2,
        cleaned_mask=clean_mask,
    )
    assert isinstance(params, ImageFitParametersContainer)


def test_truncated(prod5_lst):
    geom = prod5_lst.camera.geometry
    widths = u.Quantity([0.03, 0.05], u.m)
    lengths = u.Quantity([0.3, 0.2], u.m)
    intensity = 5000

    xs = u.Quantity([0.8, 0.7, -0.7, -0.8], u.m)
    ys = u.Quantity([-0.7, -0.8, 0.8, 0.7], u.m)

    for x, y, width, length in zip(xs, ys, widths, lengths):
        image, clean_mask = create_sample_image(
            geometry=geom, x=x, y=y, length=length, width=width, intensity=intensity
        )
        cleaned_image = image.copy()
        cleaned_image[~clean_mask] = 0.0

        # Gaussian
        result = image_fit_parameters(
            geom, image, n_row=2, cleaned_mask=clean_mask, pdf=PDFType("gaussian")
        )

        hillas = hillas_parameters(geom, cleaned_image)

        assert result.length.value > hillas.length.value

        conc_fit = concentration_parameters(geom, cleaned_image, result).core
        conc_hillas = concentration_parameters(geom, cleaned_image, hillas).core

        assert conc_fit > conc_hillas
        assert conc_fit > 0.4

        # Skewed
        result = image_fit_parameters(
            geom, image, n_row=2, cleaned_mask=clean_mask, pdf=PDFType("skewed")
        )

        assert result.length.value > hillas.length.value

        conc_fit = concentration_parameters(geom, cleaned_image, result).core

        assert conc_fit > conc_hillas
        assert conc_fit > 0.4

        # Laplace
        result = image_fit_parameters(
            geom, image, n_row=2, cleaned_mask=clean_mask, pdf=PDFType("laplace")
        )

        assert result.length.value > hillas.length.value

        conc_fit = concentration_parameters(geom, cleaned_image, result).core

        assert conc_fit > conc_hillas
        assert conc_fit > 0.4


def test_percentage(prod5_lst):
    geom = prod5_lst.camera.geometry

    widths = u.Quantity([0.01, 0.02, 0.03, 0.07], u.m)
    lengths = u.Quantity([0.1, 0.2, 0.3, 0.4], u.m)
    intensities = np.array([2000, 5000])

    xs = u.Quantity([0.1, 0.2, -0.1, -0.2], u.m)
    ys = u.Quantity([-0.2, -0.1, 0.2, 0.1], u.m)
    psis = Angle([-60, -45, 10, 45, 60], unit="deg")

    for x, y, width, length, intensity in zip(xs, ys, widths, lengths, intensities):
        for psi in psis:
            # Gaussian
            image, clean_mask = create_sample_image(psi="0d", geometry=geom)

            fit = image_fit_parameters(
                geom, image, n_row=2, cleaned_mask=clean_mask, pdf=PDFType("gaussian")
            )

            cleaned_image = image.copy()
            cleaned_image[~clean_mask] = 0.0
            conc = concentration_parameters(geom, image, fit)
            signal_inside_ellipse = conc.core

            if fit.is_valid and fit.is_accurate:
                assert signal_inside_ellipse > 0.3


def test_with_toy_mst(prod5_mst_flashcam):
    rng = np.random.default_rng(42)

    geom = prod5_mst_flashcam.camera.geometry

    width = 0.03 * u.m
    length = 0.15 * u.m
    intensity = 500

    xs = u.Quantity([0.2, 0.2, -0.2, -0.2], u.m)
    ys = u.Quantity([0.2, -0.2, 0.2, -0.2], u.m)
    psis = Angle([-60, -45, 0, 45, 60], unit="deg")

    for x, y in zip(xs, ys):
        for psi in psis:

            # make a toy shower model
            model_gaussian = toymodel.Gaussian(
                x=x, y=y, width=width, length=length, psi=psi
            )
            model_skewed = toymodel.SkewedGaussian(
                x=x, y=y, width=width, length=length, psi=psi, skewness=0.5
            )
            model_laplace = toymodel.SkewedGaussianLaplace(
                x=x, y=y, width=width, length=length, psi=psi, skewness=0.5
            )

            image, signal, noise = model_gaussian.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )

            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom,
                signal,
                n_row=0,
                cleaned_mask=clean_mask,
                pdf=PDFType("gaussian"),
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.1)
                assert u.isclose(result.y, y, rtol=0.1)

                assert u.isclose(result.width, width, rtol=0.1)
                assert u.isclose(result.length, length, rtol=0.1)
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)

            image, signal, noise = model_skewed.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )

            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom, signal, n_row=0, cleaned_mask=clean_mask, pdf=PDFType("skewed")
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.1)
                assert u.isclose(result.y, y, rtol=0.1)

                assert u.isclose(result.width, width, rtol=0.1)
                assert u.isclose(result.length, length, rtol=0.1)
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)

            image, signal, noise = model_laplace.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )
            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom, signal, n_row=0, cleaned_mask=clean_mask, pdf=PDFType("laplace")
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.1)
                assert u.isclose(result.y, y, rtol=0.1)

                assert u.isclose(result.width, width, rtol=0.15)
                assert u.isclose(result.length, length, rtol=0.1)
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)


def test_with_toy_lst(prod5_lst):
    rng = np.random.default_rng(42)

    geom = prod5_lst.camera.geometry

    width = 0.03 * u.m
    length = 0.15 * u.m
    intensity = 500

    xs = u.Quantity([0.2, 0.2, -0.2, -0.2], u.m)
    ys = u.Quantity([0.2, -0.2, 0.2, -0.2], u.m)
    psis = Angle([-60, -45, 0, 45, 60], unit="deg")

    for x, y in zip(xs, ys):
        for psi in psis:

            # make a toy shower model
            model_gaussian = toymodel.Gaussian(
                x=x, y=y, width=width, length=length, psi=psi
            )
            model_skewed = toymodel.SkewedGaussian(
                x=x, y=y, width=width, length=length, psi=psi, skewness=0.5
            )
            model_laplace = toymodel.SkewedGaussianLaplace(
                x=x, y=y, width=width, length=length, psi=psi, skewness=0.5
            )

            image, signal, noise = model_gaussian.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )

            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom,
                signal,
                n_row=0,
                cleaned_mask=clean_mask,
                pdf=PDFType("gaussian"),
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.1)
                assert u.isclose(result.y, y, rtol=0.1)

                assert u.isclose(result.width, width, rtol=0.1)
                assert u.isclose(result.length, length, rtol=0.1)
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)

            image, signal, noise = model_skewed.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )

            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom, signal, n_row=0, cleaned_mask=clean_mask, pdf=PDFType("skewed")
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.1)
                assert u.isclose(result.y, y, rtol=0.1)

                assert u.isclose(result.width, width, rtol=0.1)
                assert u.isclose(result.length, length, rtol=0.1)
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)

            image, signal, noise = model_laplace.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )
            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom, signal, n_row=0, cleaned_mask=clean_mask, pdf=PDFType("laplace")
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.1)
                assert u.isclose(result.y, y, rtol=0.1)

                assert u.isclose(result.width, width, rtol=0.15)
                assert u.isclose(result.length, length, rtol=0.1)
                assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                    result.psi.to_value(u.deg) - psi.deg
                ) == approx(180.0, abs=2)


def test_skewness(prod5_lst):
    rng = np.random.default_rng(42)

    geom = prod5_lst.camera.geometry

    widths = u.Quantity([0.02, 0.03, 0.04], u.m)
    lengths = u.Quantity([0.15, 0.2, 0.3], u.m)
    intensities = np.array([500, 1500, 2000])

    xs = u.Quantity([0.2, 0.2, -0.2, -0.2], u.m)
    ys = u.Quantity([0.2, -0.2, 0.2, -0.2], u.m)
    psis = Angle([-60, -45, 0, 45, 60, 90], unit="deg")
    skews = np.array([0, 0.5, -0.5, 0.8, -0.8, 0.9])

    for x, y, skew, intensity, width, length in zip(
        xs, ys, skews, intensities, widths, lengths
    ):
        for psi in psis:

            model_skewed = toymodel.SkewedGaussian(
                x=x, y=y, width=width, length=length, psi=psi, skewness=skew
            )
            model_laplace = toymodel.SkewedGaussianLaplace(
                x=x, y=y, width=width, length=length, psi=psi, skewness=skew
            )

            image, signal, noise = model_skewed.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )

            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom, signal, n_row=0, cleaned_mask=clean_mask, pdf=PDFType("skewed")
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.1)
                assert u.isclose(result.y, y, rtol=0.1)

                assert u.isclose(result.width, width, rtol=0.1)
                assert u.isclose(result.length, length, rtol=0.1)

                psi_same = result.psi.to_value(u.deg) == approx(psi.deg, abs=1)
                psi_opposite = abs(result.psi.to_value(u.deg) - psi.deg) == approx(
                    180.0, abs=1
                )
                assert psi_same or psi_opposite

                if psi_same:
                    assert result.skewness == approx(skew, abs=0.3)
                else:
                    assert result.skewness == approx(-skew, abs=0.3)

                assert signal.sum() == result.intensity

            image, signal, noise = model_laplace.generate_image(
                geom, intensity=intensity, nsb_level_pe=0, rng=rng
            )
            clean_mask = np.array(signal) > 0
            result = image_fit_parameters(
                geom, signal, n_row=0, cleaned_mask=clean_mask, pdf=PDFType("laplace")
            )

            if result.is_valid or result.is_accurate:
                assert u.isclose(result.x, x, rtol=0.1)
                assert u.isclose(result.y, y, rtol=0.1)

                assert u.isclose(result.width, width, rtol=0.1)
                assert u.isclose(result.length, length, rtol=0.15)

                psi_same = result.psi.to_value(u.deg) == approx(psi.deg, abs=1)
                psi_opposite = abs(result.psi.to_value(u.deg) - psi.deg) == approx(
                    180.0, abs=1
                )
                assert psi_same or psi_opposite

                if psi_same:
                    assert result.skewness == approx(skew, abs=0.3)
                else:
                    assert result.skewness == approx(-skew, abs=0.3)

                assert signal.sum() == result.intensity


def test_gaussian_skewness(prod5_lst):
    rng = np.random.default_rng(42)
    geom = prod5_lst.camera.geometry

    model_gaussian = toymodel.Gaussian(
        x=0 * u.m,
        y=0 * u.m,
        width=0.02 * u.m,
        length=0.1 * u.m,
        psi=20 * u.deg,
    )

    image, signal, noise = model_gaussian.generate_image(
        geom, intensity=1500, nsb_level_pe=0, rng=rng
    )

    clean_mask = np.array(signal) > 0
    result = image_fit_parameters(
        geom, signal, n_row=0, cleaned_mask=clean_mask, pdf=PDFType("gaussian")
    )
    assert result.skewness == 0


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
        image_fit_parameters(
            geom, image, n_row=2, cleaned_mask=clean_mask, pdf=PDFType("laplace")
        )

    with pytest.raises(ImageFitParameterizationError):
        image_fit_parameters(
            geom, image, n_row=2, cleaned_mask=clean_mask, pdf=PDFType("skewed")
        )

    with pytest.raises(ImageFitParameterizationError):
        image_fit_parameters(
            geom, image, n_row=2, cleaned_mask=clean_mask, pdf=PDFType("gaussian")
        )
