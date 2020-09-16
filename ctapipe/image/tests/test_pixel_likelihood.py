import numpy as np
from ctapipe.image import (
    neg_log_likelihood,
    neg_log_likelihood_approx,
    mean_poisson_likelihood_gaussian,
    chi_squared,
    mean_poisson_likelihood_full,
)


def test_chi_squared():
    image = np.array([20, 20, 20])
    prediction = np.array([20, 20, 20])
    bad_prediction = np.array([1, 1, 1])

    ped = 1

    chi = chi_squared(image, prediction, ped)
    bad_chi = chi_squared(image, bad_prediction, ped)

    assert chi < bad_chi


def test_mean_poisson_likelihoood_gaussian():
    prediction = np.array([1, 1, 1], dtype="float")
    spe = 0.5

    small_mean_likelihood = mean_poisson_likelihood_gaussian(prediction, spe, 0)
    large_mean_likelihood = mean_poisson_likelihood_gaussian(prediction, spe, 1)

    assert small_mean_likelihood < large_mean_likelihood


def test_mean_poisson_likelihood_full():
    prediction = np.array([30.0, 30.0])

    spe = np.array([0.5])

    small_mean_likelihood = mean_poisson_likelihood_full(prediction, spe, [0])
    large_mean_likelihood = mean_poisson_likelihood_full(prediction, spe, [1])

    assert small_mean_likelihood < large_mean_likelihood


def test_full_likelihood():
    """
    Simple test of likelihood, test against known values for high and low
    signal cases. Check that full calculation and the gaussian approx become
    equal at high signal.
    """
    spe = 0.5  # Single photo-electron width
    pedestal = 1  # width of the pedestal distribution

    image_small = np.array([0, 1, 2])
    expectation_small = np.array([1, 1, 1])

    full_like_small = neg_log_likelihood(image_small, expectation_small, spe, pedestal)
    exp_diff = full_like_small - np.sum(
        np.asarray([2.75630505, 2.62168656, 3.39248449])
    )

    # Check against known values
    assert exp_diff / np.sum(full_like_small) < 1e-4

    image_large = np.array([40, 50, 60])
    expectation_large = np.array([50, 50, 50])

    full_like_large = neg_log_likelihood(image_large, expectation_large, spe, pedestal)
    # Check against known values
    exp_diff = full_like_large - np.sum(
        np.asarray([7.45489137, 5.99305388, 7.66226007])
    )

    assert exp_diff / np.sum(full_like_large) < 1e-4

    gaus_like_large = neg_log_likelihood_approx(
        image_large, expectation_large, spe, pedestal
    )

    # Check thats in large signal case the full expectation is equal to the
    # gaussian approximation (to 5%)
    assert np.all(np.abs((full_like_large - gaus_like_large) / full_like_large) < 0.05)
