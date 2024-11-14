"""
This module contains the ctapipe.image.psf_model unit tests
"""
import numpy as np
import pytest

from ctapipe.instrument.optics import PSFModel


def test_psf(example_subarray):
    with pytest.raises(
        ValueError,
        match="asymmetry_params and az_scale_params needs to have length 3 and radial_scale_params length 4",
    ):
        PSFModel.from_name(
            "ComaModel",
            subarray=example_subarray,
            asymmetry_params=[0.49244797, 9.23573115, 0.15216096],
            radial_scale_params=[0.01409259, 0.02947208, 0.06000271, -0.02969355],
            az_scale_params=[0.24271557, 7.5511501],
        )


def test_asymptotic_behavior(example_subarray):
    psf_coma = PSFModel.from_name(
        "ComaModel",
        subarray=example_subarray,
        asymmetry_params=[0.49244797, 9.23573115, 0.15216096],
        radial_scale_params=[0.01409259, 0.02947208, 0.06000271, -0.02969355],
        az_scale_params=[0.24271557, 7.5511501, 0.02037972],
    )
    assert np.isclose(psf_coma.pdf(10.0, 0.0, 1.0, 0.0), 0.0)
