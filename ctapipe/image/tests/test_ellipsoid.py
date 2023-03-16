import itertools

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from ctapipe.containers import (CameraHillasParametersContainer,
                                HillasParametersContainer)
from ctapipe.coordinates import TelescopeFrame
from ctapipe.image import tailcuts_clean, toymodel
from ctapipe.image.ellipsoid import (ImageFitParameterizationError,
                                     image_fit_parameters)
from ctapipe.image.hillas import HillasParameterizationError, hillas_parameters
from ctapipe.instrument import CameraGeometry, SubarrayDescription
from numpy import isclose
from pytest import approx


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

def test_imagefit_failure(prod5_lst):
    geom = prod5_lst.camera.geometry
    blank_image = np.zeros(geom.n_pixels)

    with pytest.raises(ImageFitParameterizationError):
        image_fit_parameters(geom, blank_image, n=2, blank_image==1, spe_width=0.6, pesdestal=200)

def test_hillas_similarity(prod5_lst):
    geom = prod5_lst.camera.geometry
    image, clean_mask = create_sample_image(psi="0d", geometry=geom)

    cleaned_image = image.copy()
    cleaned_image[~clean_mask] = 0

    imagefit = image_fit_parameters(geom, image_zeros, n=2, clean_mask, spe_width=0.6, pedestal=200)  #TODO: include proper numbers from simulation
    hillas = hillas_parameters(geom, image_zeros)

    assert_allclose(imagefit.r, hillas.r, rtol=0.2)
    assert_allclose(imagefit.length, hillas.length, rtol=0.2)


