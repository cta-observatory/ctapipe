"""
This module contains the ctapipe.image.psf_model unit tests
"""

import astropy.units as u
import numpy as np
import pytest

from ctapipe.core.traits import TraitError
from ctapipe.instrument.optics import PSFModel


@pytest.fixture(scope="session")
def coma_psf(example_subarray):
    psf = PSFModel.from_name(
        "ComaPSFModel",
        subarray=example_subarray,
        asymmetry_params=[0.5, 10, 0.15],
        radial_scale_params=[0.015, -0.1, 0.06, 0.03],
        phi_scale_params=[0.25, 7.5, 0.02],
    )
    return psf


def test_psf(example_subarray):
    with pytest.raises(
        TraitError,
        match="phi_scale_params needs to have length 3",
    ):
        PSFModel.from_name(
            "ComaPSFModel",
            subarray=example_subarray,
            asymmetry_params=[0.0, 0.0, 0.0],
            radial_scale_params=[0.0, 0.0, 0.0, 0.0],
            phi_scale_params=[0.0],
        )
    with pytest.raises(
        TraitError,
        match="radial_scale_params needs to have length 4",
    ):
        PSFModel.from_name(
            "ComaPSFModel",
            subarray=example_subarray,
            asymmetry_params=[0.0, 0.0, 0.0],
            radial_scale_params=[0.0, 0.0, 0.0],
            phi_scale_params=[0.0, 0.0, 0.0],
        )
    with pytest.raises(
        TraitError,
        match="asymmetry_params needs to have length 3",
    ):
        PSFModel.from_name(
            "ComaPSFModel",
            subarray=example_subarray,
            asymmetry_params=[0.0, 0.0, 0.0, 0.0],
            radial_scale_params=[0.0, 0.0, 0.0, 0.0],
            phi_scale_params=[0.0, 0.0, 0.0],
        )


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
    )
