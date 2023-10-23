import numpy as np


def test_design_matrix():
    from ctapipe.fitting import design_matrix

    x = np.array([1, 2, 3])
    matrix = design_matrix(x)

    assert np.all(matrix[:, 0] == x)
    assert np.all(matrix[:, 1] == 1)


def test_linear_regression():
    from ctapipe.fitting import design_matrix, linear_regression

    # test without noise, should give exact result
    true_beta = np.array([5.0, 2.0])
    x = np.linspace(0, 10, 50)
    y = np.polyval(true_beta, x)
    X = design_matrix(x)

    beta = linear_regression(X, y)

    assert np.allclose(true_beta, beta)


def test_linear_regression_singular():
    from ctapipe.fitting import design_matrix, linear_regression

    # test under-determined input
    x = np.zeros(2, dtype=float)
    y = np.array([2, 1], dtype=float)
    X = design_matrix(x)

    beta = linear_regression(X, y)

    assert np.all(np.isnan(beta))


def test_lts_regression():
    from ctapipe.fitting import lts_linear_regression

    true_beta = np.array([5.0, 2.0])

    # test without noise, should give exact result
    x = np.linspace(0, 10, 50)
    y = np.polyval(true_beta, x)

    beta, error = lts_linear_regression(x, y)

    assert np.allclose(true_beta, beta)
    assert np.isclose(error, 0)

    # add a single outlier, it should give the same result
    y[-1] *= 3

    beta, error = lts_linear_regression(x, y)

    # make sure the outlier effects normal linear regression
    assert not np.allclose(true_beta, np.polyfit(x, y, deg=1))
    assert np.allclose(true_beta, beta)
    assert np.isclose(error, 0)

    # add noise, should still give a good result, but not exact
    rng = np.random.default_rng(1337)
    y += rng.normal(0, 0.1, len(y))

    beta, error = lts_linear_regression(x, y)

    # larger rtol since we added noise
    assert np.allclose(true_beta, beta, rtol=0.05)


def test_lts_regression_singular_pair():
    """
    Test the lts regression with data that contains a pair
    of points creating a singular matrix
    """
    from ctapipe.fitting import EPS, design_matrix, lts_linear_regression

    true_beta = np.array([5.0, 2.0])

    # make test contain data that creates a singular matrix
    x = np.linspace(0, 10, 25)
    x = np.repeat(x, 2)
    y = np.polyval(true_beta, x)

    x[:2] = 0
    y[:2] = 1

    X = design_matrix(x[:2])
    assert np.linalg.det(X.T @ X) < EPS

    beta, error = lts_linear_regression(x, y, samples=100)

    assert np.allclose(true_beta, beta)
    assert np.isclose(error, 0)
