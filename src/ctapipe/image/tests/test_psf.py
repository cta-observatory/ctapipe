from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest

from ctapipe.image.psf_model import ComaModel
from ctapipe.instrument import SubarrayDescription


@pytest.fixture(scope="module")
def subarray(prod5_sst, reference_location):
    subarray = SubarrayDescription(
        "test array",
        tel_positions={1: np.zeros(3) * u.m, 2: np.zeros(3) * u.m},
        tel_descriptions={
            1: deepcopy(prod5_sst),
            2: deepcopy(prod5_sst),
        },
        reference_location=reference_location,
    )
    return subarray


psf = ComaModel(subarray=subarray)
psf.update_model_parameters(0.01, 0.0)
print(psf.pdf(0.01, 0.0))
