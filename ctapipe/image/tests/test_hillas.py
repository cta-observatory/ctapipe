from ctapipe.instrument import CameraGeometry
from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.containers import HillasParametersContainer
from astropy.coordinates import Angle
from astropy import units as u
import numpy as np
from numpy import isclose, zeros_like
from pytest import approx
import itertools
import pytest


def create_sample_image(
    psi="-30d",
    x=0.2 * u.m,
    y=0.3 * u.m,
    width=0.05 * u.m,
    length=0.15 * u.m,
    intensity=1500,
):

    geom = CameraGeometry.from_name("LSTCam")

    # make a toymodel shower model
    model = toymodel.Gaussian(x=x, y=y, width=width, length=length, psi=psi)

    # generate toymodel image in camera for this shower model.
    rng = np.random.default_rng(0)
    image, _, _ = model.generate_image(geom, intensity=1500, nsb_level_pe=3, rng=rng)

    # calculate pixels likely containing signal
    clean_mask = tailcuts_clean(geom, image, 10, 5)

    return geom, image, clean_mask


def create_sample_image_zeros(psi="-30d"):

    geom, image, clean_mask = create_sample_image(psi)

    # threshold in pe
    image[~clean_mask] = 0

    return geom, image


def create_sample_image_selected_pixel(psi="-30d"):
    geom, image, clean_mask = create_sample_image(psi)

    return geom[clean_mask], image[clean_mask]


def compare_result(x, y):
    ux = u.Quantity(x)
    uy = u.Quantity(y)
    assert isclose(ux.value, uy.value)
    assert ux.unit == uy.unit


def compare_hillas(hillas1, hillas2):
    hillas1_dict = hillas1.as_dict()
    hillas2_dict = hillas2.as_dict()
    for key in hillas1_dict.keys():
        compare_result(hillas1_dict[key], hillas2_dict[key])


def test_hillas_selected():
    """
    test Hillas-parameter routines on a sample image with selected values
    against a sample image with masked values set tozero
    """
    geom, image = create_sample_image_zeros()
    geom_selected, image_selected = create_sample_image_selected_pixel()

    results = hillas_parameters(geom, image)
    results_selected = hillas_parameters(geom_selected, image_selected)

    compare_hillas(results, results_selected)


def test_hillas_failure():
    geom, image = create_sample_image_zeros(psi="0d")
    blank_image = zeros_like(image)

    with pytest.raises(HillasParameterizationError):
        hillas_parameters(geom, blank_image)


def test_hillas_masked_array():
    geom, image, clean_mask = create_sample_image(psi="0d")

    image_zeros = image.copy()
    image_zeros[~clean_mask] = 0
    hillas_zeros = hillas_parameters(geom, image_zeros)

    image_masked = np.ma.masked_array(image, mask=~clean_mask)
    hillas_masked = hillas_parameters(geom, image_masked)

    compare_hillas(hillas_zeros, hillas_masked)


def test_hillas_container():
    geom, image = create_sample_image_zeros(psi="0d")

    params = hillas_parameters(geom, image)
    assert isinstance(params, HillasParametersContainer)


def test_with_toy():
    rng = np.random.default_rng(42)

    geom = CameraGeometry.from_name("LSTCam")

    width = 0.03 * u.m
    length = 0.15 * u.m
    width_uncertainty = 0.00094 * u.m
    length_uncertainty = 0.00465 * u.m
    intensity = 500

    xs = u.Quantity([0.5, 0.5, -0.5, -0.5], u.m)
    ys = u.Quantity([0.5, -0.5, 0.5, -0.5], u.m)
    psis = Angle([-90, -45, 0, 45, 90], unit="deg")

    for x, y in zip(xs, ys):
        for psi in psis:

            # make a toymodel shower model
            model = toymodel.Gaussian(x=x, y=y, width=width, length=length, psi=psi)

            image, signal, noise = model.generate_image(
                geom, intensity=intensity, nsb_level_pe=5, rng=rng
            )

            result = hillas_parameters(geom, signal)

            assert u.isclose(result.x, x, rtol=0.1)
            assert u.isclose(result.y, y, rtol=0.1)

            assert u.isclose(result.width, width, rtol=0.1)
            assert u.isclose(result.width_uncertainty, width_uncertainty, rtol=0.4)
            assert u.isclose(result.length, length, rtol=0.1)
            assert u.isclose(result.length_uncertainty, length_uncertainty, rtol=0.4)
            assert (result.psi.to_value(u.deg) == approx(psi.deg, abs=2)) or abs(
                result.psi.to_value(u.deg) - psi.deg
            ) == approx(180.0, abs=2)

            assert signal.sum() == result.intensity


def test_skewness():
    rng = np.random.default_rng(42)

    geom = CameraGeometry.from_name("LSTCam")

    width = 0.03 * u.m
    length = 0.15 * u.m
    intensity = 2500

    xs = u.Quantity([0.5, 0.5, -0.5, -0.5], u.m)
    ys = u.Quantity([0.5, -0.5, 0.5, -0.5], u.m)
    psis = Angle([-90, -45, 0, 45, 90], unit="deg")
    skews = [0, 0.3, 0.6]

    for x, y, psi, skew in itertools.product(xs, ys, psis, skews):
        # make a toymodel shower model
        model = toymodel.SkewedGaussian(
            x=x, y=y, width=width, length=length, psi=psi, skewness=skew
        )

        _, signal, _ = model.generate_image(
            geom, intensity=intensity, nsb_level_pe=5, rng=rng
        )

        result = hillas_parameters(geom, signal)

        assert u.isclose(result.x, x, rtol=0.1)
        assert u.isclose(result.y, y, rtol=0.1)

        assert u.isclose(result.width, width, rtol=0.1)
        assert u.isclose(result.length, length, rtol=0.1)

        psi_same = result.psi.to_value(u.deg) == approx(psi.deg, abs=3)
        psi_opposite = abs(result.psi.to_value(u.deg) - psi.deg) == approx(180.0, abs=3)
        assert psi_same or psi_opposite

        # if we have delta the other way around, we get a negative sign for skewness
        # skewness is quite imprecise, maybe we could improve this somehow
        if psi_same:
            assert result.skewness == approx(skew, abs=0.3)
        else:
            assert result.skewness == approx(-skew, abs=0.3)

        assert signal.sum() == result.intensity


@pytest.mark.filterwarnings("error")
def test_straight_line_width_0():
    """ Test that hillas_parameters.width is 0 for a straight line of pixels """
    # three pixels in a straight line
    long = np.array([0, 1, 2]) * 0.01
    trans = np.zeros(len(long))
    pix_id = np.arange(len(long))

    rng = np.random.default_rng(0)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for psi in np.linspace(0, np.pi, 20):
                x = dx + np.cos(psi) * long + np.sin(psi) * trans
                y = dy - np.sin(psi) * long + np.cos(psi) * trans

                geom = CameraGeometry(
                    camera_name="testcam",
                    pix_id=pix_id,
                    pix_x=x * u.m,
                    pix_y=y * u.m,
                    pix_type="hexagonal",
                    pix_area=1 * u.m ** 2,
                )

                img = rng.poisson(5, size=len(long))
                result = hillas_parameters(geom, img)
                assert result.width.value == 0
                assert np.isnan(result.width_uncertainty.value)


@pytest.mark.filterwarnings("error")
def test_single_pixel():
    x = y = np.arange(3)
    x, y = np.meshgrid(x, y)

    geom = CameraGeometry(
        camera_name="testcam",
        pix_id=np.arange(9),
        pix_x=x.ravel() * u.cm,
        pix_y=y.ravel() * u.cm,
        pix_type="rectangular",
        pix_area=1 * u.cm ** 2,
    )

    image = np.zeros((3, 3))
    image[1, 1] = 10
    image = image.ravel()

    hillas = hillas_parameters(geom, image)

    assert hillas.length.value == 0
    assert hillas.width.value == 0
    assert np.isnan(hillas.psi)
