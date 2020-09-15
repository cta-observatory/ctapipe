import numpy as np


def test_design_matrix():
    from ctapipe.fitting import design_matrix

    x = np.array([1, 2, 3])
    matrix = design_matrix(x)

    assert np.all(matrix[:, 0] == x)
    assert np.all(matrix[:, 1] == 1)


def test_lts_regression():
    from ctapipe.fitting import lts_linear_regression

    true_beta = np.array([5.0, 2.0])

    # test without noise, should give exact result
    x = np.linspace(0, 10, 20)
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
    y += np.random.normal(0, 0.1, len(y))

    beta, error = lts_linear_regression(x, y)

    # larger rtol since we added noise
    assert np.allclose(true_beta, beta, rtol=0.05)
