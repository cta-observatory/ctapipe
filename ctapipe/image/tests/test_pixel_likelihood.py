import numpy as np
from ctapipe.image import poisson_likelihood_full, poisson_likelihood_gaussian
import pytest

@pytest.mark.skip   # skip until somebody can find out why it is wrong
def test_full_likelihood():
    """
    Simple test of likelihood, test against known values for high and low 
    signal cases. Check that full calculation and the gaussian approx become 
    equal at high signal.
    """
    spe = 0.5 # Single photo-electron width
    pedestal = 1 # width of the pedestal distribution

    image_small = [0,1,2]
    expectation_small = [1,1,1]

    full_like_small = poisson_likelihood_full(image_small, expectation_small,
                                              spe, pedestal)
    # Check against known values
    assert np.sum(np.abs(full_like_small - np.asarray([2.75630505, 2.62168656,
                                                       3.39248449]))) < 1e-6

    image_large = [40,50,60]
    expectation_large = [50,50,50]
    full_like_large = poisson_likelihood_full(image_large, expectation_large,
                                              spe, pedestal)
    # Check against known values
    assert np.sum(np.abs(full_like_large - np.asarray([7.45489137,
                                                       5.99305388,
                                                       7.66226007]))) < 1e-6

    gaus_like_large = poisson_likelihood_gaussian(image_large,
                                                  expectation_large,
                                                  spe,
                                                  pedestal)

    # Check thats in large signal case the full expectation is equal to the
    # gaussian approximation (to 5%)
    assert np.all(np.abs((full_like_large-gaus_like_large)/full_like_large)
                  < 0.05)
