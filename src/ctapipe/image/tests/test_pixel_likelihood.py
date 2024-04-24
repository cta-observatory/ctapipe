import numpy as np

from ctapipe.image import (
    chi_squared,
    mean_poisson_likelihood_full,
    mean_poisson_likelihood_gaussian,
    neg_log_likelihood,
    neg_log_likelihood_approx,
    neg_log_likelihood_numeric,
)


def test_chi_squared():
    image = np.array([20, 20, 20])
    prediction = np.array([20, 20, 20])
    bad_prediction = np.array([1, 1, 1])

    ped = 1

    chi = chi_squared(image, prediction, ped)
    bad_chi = chi_squared(image, bad_prediction, ped)

    assert np.sum(chi) < np.sum(bad_chi)


def test_mean_poisson_likelihoood_gaussian():
    prediction = np.array([50, 50, 50], dtype="float")
    spe = 0.5

    small_mean_likelihood = mean_poisson_likelihood_gaussian(prediction, spe, 0)
    large_mean_likelihood = mean_poisson_likelihood_gaussian(prediction, spe, 1)

    assert np.all(small_mean_likelihood < large_mean_likelihood)

    # Test that the mean likelihood of abunch of samples drawn from the gaussian
    # behind the approximate log likelihood is indeed the precalculated mean

    rng = np.random.default_rng(123456)

    ped = 1

    mean_likelihood = mean_poisson_likelihood_gaussian(prediction[0], spe, ped)

    distribution_width = np.sqrt(ped**2 + prediction[0] * (1 + spe**2))

    normal_samples = rng.normal(
        loc=prediction[0], scale=distribution_width, size=100000
    )

    rel_diff = (
        np.mean(2 * neg_log_likelihood_approx(normal_samples, prediction[0], spe, ped))
        - mean_likelihood
    ) / mean_likelihood

    assert np.abs(rel_diff) < 5e-4


def test_mean_poisson_likelihood_full():
    prediction = np.array([10.0, 10.0])

    spe = np.array([0.5])

    small_mean_likelihood = mean_poisson_likelihood_full(prediction, spe, [0.1])
    large_mean_likelihood = mean_poisson_likelihood_full(prediction, spe, [1])

    assert np.all(small_mean_likelihood < large_mean_likelihood)


def test_full_likelihood():
    """
    Simple test of likelihood, test against known values for high and low
    signal cases. Check that full calculation and the gaussian approx become
    equal at high signal.
    """
    spe = 0.5 * np.ones(3)  # Single photo-electron width
    pedestal = np.ones(3)  # width of the pedestal distribution

    image_small = np.array([0, 1, 2])
    expectation_small = np.array([1, 1, 1])

    full_like_small = neg_log_likelihood(image_small, expectation_small, spe, pedestal)
    exp_rel_diff = (
        full_like_small - np.asarray([1.37815294, 1.31084662, 1.69627197])
    ) / full_like_small

    # Check against known values
    assert np.all(np.abs(exp_rel_diff) < 3e-4)

    image_large = np.array([40, 50, 60])
    expectation_large = np.array([50, 50, 50])

    full_like_large = neg_log_likelihood(image_large, expectation_large, spe, pedestal)
    # Check against known values
    exp_rel_diff = (
        full_like_large - np.asarray([3.78183004, 2.99452694, 3.78183004])
    ) / full_like_large

    assert np.all(np.abs(exp_rel_diff) < 3e-5)

    gaus_like_large = neg_log_likelihood_approx(
        image_large, expectation_large, spe, pedestal
    )

    numeric_like_large = neg_log_likelihood_numeric(
        image_large, expectation_large, spe, pedestal
    )

    # Check that in the large signal case the full expectation is equal to the
    # gaussian approximation (to 5%)
    assert np.all(
        np.abs((numeric_like_large - gaus_like_large) / numeric_like_large) < 0.05
    )
