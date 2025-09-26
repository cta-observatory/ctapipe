from collections import namedtuple

import astropy.units as u
import numpy as np
import pytest
from scipy.constants import alpha

parameter_names = [
    "radius",
    "rho",
    "phi",
    "expected_length",
]
Parameters = namedtuple("MuonTestParams", parameter_names)


@pytest.mark.parametrize(
    parameter_names,
    [
        Parameters(
            radius=12,
            rho=0.0,
            phi=0.0 * u.deg,
            expected_length=12,
        ),
        Parameters(
            radius=12,
            rho=12,
            phi=90.0 * u.deg,
            expected_length=0,
        ),
        Parameters(
            radius=12,
            rho=13,
            phi=180.0 * u.deg,
            expected_length=0,
        ),
        Parameters(
            radius=12,
            rho=24.0,
            phi=0.0 * u.deg,
            expected_length=24,
        ),
    ],
)
def test_chord_length(
    radius,
    rho,
    phi,
    expected_length,
):
    from ctapipe.image.muon.intensity_fitter import chord_length

    length = chord_length(radius, rho, phi.to_value(u.rad))
    assert np.isclose(length, expected_length, atol=1e-15)


def test_muon_efficiency_fit(prod5_lst, reference_location):
    from ctapipe.coordinates import TelescopeFrame
    from ctapipe.image.muon.intensity_fitter import (
        MuonIntensityFitter,
        image_prediction,
    )
    from ctapipe.instrument import SubarrayDescription

    pytest.importorskip("iminuit")

    tel_id = 1
    telescope = prod5_lst
    subarray = SubarrayDescription(
        name="LSTMono",
        tel_positions={tel_id: [0, 0, 0] * u.m},
        tel_descriptions={tel_id: telescope},
        reference_location=reference_location,
    )

    center_x = 0.8 * u.deg
    center_y = 0.4 * u.deg
    radius = 1.1 * u.deg
    ring_width = 0.05 * u.deg
    impact_parameter = 5 * u.m
    phi = 0 * u.rad
    efficiency = 0.5

    geom = telescope.camera.geometry.transform_to(TelescopeFrame())
    mirror_radius = np.sqrt(telescope.optics.mirror_area / np.pi)

    pixel_diameter = geom.pixel_width[0]
    x = geom.pix_x
    y = geom.pix_y

    fitter = MuonIntensityFitter(subarray=subarray)

    image = image_prediction(
        mirror_radius,
        hole_radius=fitter.hole_radius_m.tel[tel_id] * u.m,
        impact_parameter=impact_parameter,
        phi=phi,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        ring_width=ring_width,
        pixel_x=x,
        pixel_y=y,
        pixel_diameter=pixel_diameter,
        pix_type=telescope.camera.geometry.pix_type,
    )

    result = fitter(
        tel_id=tel_id,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        image=image * efficiency,
        pedestal=np.full_like(image, 1.1),
    )

    assert u.isclose(result.impact, impact_parameter, rtol=0.05)
    assert u.isclose(result.width, ring_width, rtol=0.05)
    assert u.isclose(result.optical_efficiency, efficiency, rtol=0.05)
    assert result.is_valid
    assert not result.parameters_at_limit
    assert np.isfinite(result.likelihood_value)


def test_scts(prod5_sst, reference_location):
    from ctapipe.image.muon.intensity_fitter import MuonIntensityFitter
    from ctapipe.instrument import SubarrayDescription

    pytest.importorskip("iminuit")

    telescope = prod5_sst
    subarray = SubarrayDescription(
        name="ssts",
        tel_positions={0: [0, 0, 0] * u.m},
        tel_descriptions={0: telescope},
        reference_location=reference_location,
    )

    fitter = MuonIntensityFitter(subarray=subarray)
    with pytest.raises(NotImplementedError):
        fitter(
            tel_id=0,
            center_x=0 * u.deg,
            center_y=2 * u.deg,
            radius=1.3 * u.deg,
            image=np.zeros(telescope.camera.geometry.n_pixels),
            pedestal=np.zeros(telescope.camera.geometry.n_pixels),
        )


def test_normalisation_factor(prod5_lst, reference_location):
    """Test of the absolute normalization factor."""
    from ctapipe.coordinates import TelescopeFrame
    from ctapipe.image.muon.intensity_fitter import (
        image_prediction,
    )

    pytest.importorskip("iminuit")

    telescope = prod5_lst

    geom = telescope.camera.geometry.transform_to(TelescopeFrame())
    mirror_radius = np.sqrt(telescope.optics.mirror_area / np.pi)

    pixel_diameter = geom.pixel_width[0]
    x = geom.pix_x
    y = geom.pix_y

    image = image_prediction(
        mirror_radius,
        hole_radius=0 * u.m,
        impact_parameter=0 * u.m,
        phi=0 * u.rad,
        center_x=0.0 * u.deg,
        center_y=0.0 * u.deg,
        radius=1.1 * u.deg,
        ring_width=0.05 * u.deg,
        pixel_x=x,
        pixel_y=y,
        pixel_diameter=pixel_diameter,
        oversampling=3,
        min_lambda=300 * u.nm,
        max_lambda=600 * u.nm,
        pix_type=telescope.camera.geometry.pix_type,
    )

    measured = np.sum(image)
    expected = expected_nphot(
        Rmirror=mirror_radius,
        theta_cher=1.1 * u.deg,
        lambda_min=300 * u.nm,
        lambda_max=600 * u.nm,
    )

    assert u.isclose(measured, expected, rtol=0.02)


def expected_nphot(Rmirror, theta_cher, lambda_min, lambda_max):
    """
    The trivial solution for the number of photons incident on the telescope mirror.

    It is a trivial case, since we assume a muon impact at the center of the dish,
    with no shadowing and a constant Cherenkov angle.
    We neglect the light yield attenuation due to atmospheric absorption.

    Parameters
    ----------
    Rmirror: quantity[length]
        mirror radius
    theta_cher: quantity[angle]
        Cherenkov angle
    lambda_min: quantity[length]
        photon wavelength
    lambda_max: quantity[length]
        photon wavelength

    Returns
    -------
    float: number of Cherenkov photons

    """

    return (
        np.pi
        * alpha
        * Rmirror.to_value(u.m)
        * np.sin(2 * theta_cher)
        * (lambda_min.to_value(u.m) ** -1 - lambda_max.to_value(u.m) ** -1)
    )
