"""
This module contains the ctapipe.image.psf_model unit tests
"""
import numpy as np
import pytest

from ctapipe.instrument.optics import PSFModel


@pytest.fixture(scope="session")
def asymmetry_params():
    return [0.49244797, 9.23573115, 0.15216096]


@pytest.fixture(scope="session")
def radial_scale_params():
    return [0.01409259, -0.02947208, 0.06000271, -0.02969355]


@pytest.fixture(scope="session")
def az_scale_params():
    return [0.24271557, 7.5511501, 0.02037972]


def test_psf(example_subarray, asymmetry_params, radial_scale_params):
    with pytest.raises(
        ValueError,
        match="asymmetry_params and az_scale_params needs to have length 3 and radial_scale_params length 4",
    ):
        PSFModel.from_name(
            "ComaModel",
            subarray=example_subarray,
            asymmetry_params=asymmetry_params,
            radial_scale_params=radial_scale_params,
            az_scale_params=[0.0],
        )


def test_asymptotic_behavior(
    example_subarray, asymmetry_params, radial_scale_params, az_scale_params
):
    psf_coma = PSFModel.from_name(
        "ComaModel",
        subarray=example_subarray,
        asymmetry_params=asymmetry_params,
        radial_scale_params=radial_scale_params,
        az_scale_params=az_scale_params,
    )
    assert np.isclose(psf_coma.pdf(10.0, 0.0, 1.0, 0.0), 0.0)
