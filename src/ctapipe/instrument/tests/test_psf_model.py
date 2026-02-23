"""
This module contains the ctapipe.image.psf_model unit tests
"""

import astropy.units as u
import numpy as np
import pytest

from ctapipe.instrument.optics import PSFModel


@pytest.fixture(scope="session")
def coma_psf(example_subarray):
    psf = PSFModel.from_name(
        "ComaPSFModel",
        subarray=example_subarray,
        asymmetry_max=0.5,
        asymmetry_decay_rate=10,
        asymmetry_linear_term=0.15,
        radial_scale_offset=0.015,
        radial_scale_linear=-0.1,
        radial_scale_quadratic=0.06,
        radial_scale_cubic=0.03,
        polar_scale_amplitude=0.25,
        polar_scale_decay=7.5,
        polar_scale_offset=0.02,
    )
    return psf


def test_asymptotic_behavior(coma_psf):
    assert np.isclose(
        coma_psf.pdf(
            tel_id=1,
            lon=20.0 * u.deg,
            lat=0.0 * u.deg,
            lon0=2.0 * u.deg,
            lat0=0.0 * u.deg,
        ),
        0.0,
        atol=1e-7,
    )


def test_for_missing_config_parameters(example_subarray):
    with pytest.raises(
        ValueError, match="Missing ComaPSFModel configuration parameters:"
    ):
        PSFModel.from_name(
            "ComaPSFModel",
            subarray=example_subarray,
        )
