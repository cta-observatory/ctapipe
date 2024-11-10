"""
This module contains the ctapipe.image.psf_model unit tests
"""
import numpy as np
import pytest

from ctapipe.image.psf_model import PSFModel


def test_psf():
    psf = PSFModel.from_name("ComaModel")
    wrong_parameters = {
        "asymmetry_params": [0.49244797, 9.23573115, 0.15216096],
        "radial_scale_params": [0.01409259, 0.02947208, 0.06000271, -0.02969355],
        "az_scale_params": [0.24271557, 7.5511501],
    }
    with pytest.raises(
        ValueError,
        match="asymmetry_params and az_scale_params needs to have length 3 and radial_scale_params length 4",
    ):
        psf.update_model_parameters(wrong_parameters)


def test_asymptotic_behavior():
    psf_coma = PSFModel.from_name("ComaModel")
    psf_coma.update_location(1.0, 0.0)
    assert np.isclose(psf_coma.pdf(10.0, 0.0), 0.0)
