import itertools

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from numpy import isclose
from pytest import approx

from ctapipe.containers import (
    CameraHillasParametersContainer,
    HillasParametersContainer,
)
from ctapipe.coordinates import TelescopeFrame
from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.hillas import HillasParameterizationError, hillas_parameters
from ctapipe.instrument import CameraGeometry, SubarrayDescription


def create_sample_image(
    psi="-30d",
    x=0.2 * u.m,
    y=0.3 * u.m,
    width=0.05 * u.m,
    length=0.15 * u.m,
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
    image, _, _ = model.generate_image(
        geometry, intensity=intensity, nsb_level_pe=3, rng=rng
    )

    # calculate pixels likely containing signal
    clean_mask = tailcuts_clean(geometry, image, 10, 5)

    return image, clean_mask


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


def test_hillas_selected(prod5_lst):
    """
    test Hillas-parameter routines on a sample image with selected values
    against a sample image with masked values set to zero
    """
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(geometry=geom)

    image_zeros = image.copy()
    image_zeros[~clean_mask] = 0.0

    image_selected = image[clean_mask]
    geom_selected = geom[clean_mask]

    results = hillas_parameters(geom, image_zeros)
    results_selected = hillas_parameters(geom_selected, image_selected)

    compare_hillas(results, results_selected)


def test_hillas_failure(prod5_lst):
    geom = prod5_lst.camera.geometry
    blank_image = np.zeros(geom.n_pixels)

    with pytest.raises(HillasParameterizationError):
        hillas_parameters(geom, blank_image)


def test_hillas_masked_array(prod5_lst):
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(psi="0d", geometry=geom)

    image_zeros = image.copy()
    image_zeros[~clean_mask] = 0
    hillas_zeros = hillas_parameters(geom, image_zeros)

    image_masked = np.ma.masked_array(image, mask=~clean_mask)
    hillas_masked = hillas_parameters(geom, image_masked)

    compare_hillas(hillas_zeros, hillas_masked)


def test_hillas_container(prod5_lst):
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(psi="0d", geometry=geom)

    params = hillas_parameters(geom[clean_mask], image[clean_mask])
    assert isinstance(params, CameraHillasParametersContainer)

    geom_telescope_frame = geom.transform_to(TelescopeFrame())
    params = hillas_parameters(geom_telescope_frame[clean_mask], image[clean_mask])
    assert isinstance(params, HillasParametersContainer)


def test_with_toy(prod5_lst):
    rng = np.random.default_rng(42)

    geom = prod5_lst.camera.geometry

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


def test_skewness(prod5_lst):
    rng = np.random.default_rng(42)

    geom = prod5_lst.camera.geometry

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
    """Test that hillas_parameters.width is 0 for a straight line of pixels"""
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
                    name="testcam",
                    pix_id=pix_id,
                    pix_x=x * u.m,
                    pix_y=y * u.m,
                    pix_type="hexagonal",
                    pix_area=np.full(len(pix_id), 1.0) * u.m**2,
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
        name="testcam",
        pix_id=np.arange(9),
        pix_x=x.ravel() * u.cm,
        pix_y=y.ravel() * u.cm,
        pix_type="rectangular",
        pix_area=np.full(9, 1.0) * u.cm**2,
    )

    image = np.zeros((3, 3))
    image[1, 1] = 10
    image = image.ravel()

    hillas = hillas_parameters(geom, image)

    assert hillas.length.value == 0
    assert hillas.width.value == 0
    assert np.isnan(hillas.psi)


def test_reconstruction_in_telescope_frame(prod5_lst):
    """
    Compare the reconstruction in the telescope
    and camera frame.
    """
    np.random.seed(42)

    geom = prod5_lst.camera.geometry
    telescope_frame = TelescopeFrame()
    camera_frame = geom.frame
    geom_nom = geom.transform_to(telescope_frame)

    width = 0.03 * u.m
    length = 0.15 * u.m
    intensity = 500

    xs = u.Quantity([0.5, 0.5, -0.5, -0.5], u.m)
    ys = u.Quantity([0.5, -0.5, 0.5, -0.5], u.m)
    psis = Angle([-90, -45, 0, 45, 90], unit="deg")

    def distance(coord):
        return np.sqrt(np.diff(coord.x) ** 2 + np.diff(coord.y) ** 2) / 2

    def get_transformed_length(telescope_hillas, telescope_frame, camera_frame):
        main_edges = u.Quantity([-telescope_hillas.length, telescope_hillas.length])
        main_lon = main_edges * np.cos(telescope_hillas.psi) + telescope_hillas.fov_lon
        main_lat = main_edges * np.sin(telescope_hillas.psi) + telescope_hillas.fov_lat
        cam_main_axis = SkyCoord(
            fov_lon=main_lon, fov_lat=main_lat, frame=telescope_frame
        ).transform_to(camera_frame)
        transformed_length = distance(cam_main_axis)
        return transformed_length

    def get_transformed_width(telescope_hillas, telescope_frame, camera_frame):
        secondary_edges = u.Quantity([-telescope_hillas.width, telescope_hillas.width])
        secondary_lon = (
            secondary_edges * np.cos(telescope_hillas.psi) + telescope_result.fov_lon
        )
        secondary_lat = (
            secondary_edges * np.sin(telescope_hillas.psi) + telescope_result.fov_lat
        )
        cam_secondary_edges = SkyCoord(
            fov_lon=secondary_lon, fov_lat=secondary_lat, frame=telescope_frame
        ).transform_to(camera_frame)
        transformed_width = distance(cam_secondary_edges)
        return transformed_width

    for x, y in zip(xs, ys):
        for psi in psis:
            # generate a toy image
            model = toymodel.Gaussian(x=x, y=y, width=width, length=length, psi=psi)
            image, signal, noise = model.generate_image(
                geom, intensity=intensity, nsb_level_pe=5
            )

            telescope_result = hillas_parameters(geom_nom, signal)
            camera_result = hillas_parameters(geom, signal)
            assert camera_result.intensity == telescope_result.intensity

            # Compare results in both frames
            transformed_cog = SkyCoord(
                fov_lon=telescope_result.fov_lon,
                fov_lat=telescope_result.fov_lat,
                frame=telescope_frame,
            ).transform_to(camera_frame)
            assert u.isclose(transformed_cog.x, camera_result.x, rtol=0.01)
            assert u.isclose(transformed_cog.y, camera_result.y, rtol=0.01)

            transformed_length = get_transformed_length(
                telescope_result, telescope_frame, camera_frame
            )
            assert u.isclose(transformed_length, camera_result.length, rtol=0.01)

            transformed_width = get_transformed_width(
                telescope_result, telescope_frame, camera_frame
            )
            assert u.isclose(transformed_width, camera_result.width, rtol=0.01)
