import numpy as np
from numba import njit

from .core.env import CTAPIPE_DISABLE_NUMBA_CACHE

EPS = 2 * np.finfo(np.float64).eps
FIT_RNG = np.random.default_rng(0)


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
def design_matrix(x):
    """
    Build the design matrix for linear regression for
    a given array of x-values.

    This creates a (N, 2) matrix, where the first column are the
    x values and the second contains all 1.
    """
    X = np.empty((len(x), 2), dtype=x.dtype)
    X[:, 0] = x
    X[:, 1] = 1

    return X


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
def linear_regression(X, y):
    """
    Analytical linear regression

    Arguments
    ---------
    X: np.array
        Design matrix of shape (N, 2), as created by ``~ctapipe.fitting.design_matrix``
    y: np.array
        y values
    """
    mat = X.T @ X
    if np.linalg.det(mat) < EPS:
        return np.full(2, np.nan)

    return np.linalg.inv(mat) @ X.T @ y


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
def residual_sum_of_squares(X, y, beta):
    """Calculate the residual sum of squares

    Arguments
    ---------
    X: np.array
        Design matrix of shape (N, 2), as created by ``~ctapipe.fitting.design_matrix``
    y: np.array
        y values
    beta: np.array
        Parameter vector of the linear regression
    """
    return np.sum(residuals(X, y, beta) ** 2)


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
def residuals(X, y, beta):
    """Calculate the residuals of a linear regression

    Arguments
    ---------
    X: np.array
        Design matrix of shape (N, 2), as created by ``~ctapipe.fitting.design_matrix``
    y: np.array
        y values
    beta: np.array
        Parameter vector of the linear regression
    """
    return y - (X[:, 0] * beta[0] + beta[1])


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
def choose_two_without_replacement(rng, n):
    values = np.empty(2, dtype=np.int64)

    # we need to draw two distinct integers in [0, n)
    # first sample can just be draw uniform:
    values[0] = rng.integers(0, n)
    # we draw the second sample in 0, n - 1 and shift by 1
    # if it's >= the first value, this means we draw uniform
    # in [0, ) with the exception of the first value
    values[1] = rng.integers(0, n - 1)
    if values[1] >= values[0]:
        values[1] += 1
    return values


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
def _lts_single_sample(X, y, sample_size, max_iterations, rng, eps=1e-12):
    # randomly draw 2 points for the initial fit
    sample = choose_two_without_replacement(rng, len(y))

    # perform the initial fit
    beta = linear_regression(X[sample], y[sample])
    if np.isnan(beta[0]):
        return beta, np.nan

    error = residual_sum_of_squares(X[sample], y[sample], beta)

    for i in range(max_iterations):
        squared_residuals = residuals(X, y, beta) ** 2

        # select the subset with the smallest squared residuals
        sample = np.argsort(squared_residuals)[:sample_size]

        # redo regression with new sample
        beta = linear_regression(X[sample], y[sample])

        last_error = error
        error = residual_sum_of_squares(X[sample], y[sample], beta)

        # test for convergences
        if abs(last_error - error) < eps:
            break

    return beta, error


def lts_linear_regression(
    x,
    y,
    samples=20,
    relative_sample_size=0.85,
    max_iterations=20,
    eps=1e-12,
    rng=FIT_RNG,
):
    """
    Perform a Least Trimmed Squares regression based on algorithm (2) described in [lts_regression]_

    We start from randomly sampled two points of the dataset and then
    iteratively choose the `relative_sample_size` fraction of points with the smallest
    residuals to redo the fit until it convergtes.

    This is done for ``samples`` initial samples and the solution with the
    lowest residual sum of squares is returned along with that error.

    Arguments
    ---------
    x: np.array
        x values for the linear regression
    y: np.array
        y values for the linear regression
    samples: int
        How many initial samples to generate.
        For each sample, the C-step optimization is performed and the result
        with the lowest sum of squares is returned.
    relative_sample_size: float
        Portion of points to be used for the fit, should be > 0.5
    max_iterations: int
        maximum number of C-Steps performed for each sample.
        The fit should converge much faster (normally after < 10 iterations)
    eps: float
        differences in residual sum of squares at which the iteration will be stopped
    rng: numpy.random.Generator
        Random generator for drawing fit samples

    Returns
    -------
    beta: np.array
        Parameter vector of the linear regression
    error: float
        Residual sum of squares of the best fit
    """
    # numba currently does not support passing rng via default, so we have a
    # pure python wrapper that calls the numba jitted function with he defaults filled in
    return _lts_linear_regression(
        x,
        y,
        rng,
        samples=samples,
        relative_sample_size=relative_sample_size,
        max_iterations=max_iterations,
        eps=eps,
    )


@njit(cache=not CTAPIPE_DISABLE_NUMBA_CACHE)
def _lts_linear_regression(
    x,
    y,
    rng,
    samples,
    relative_sample_size,
    max_iterations,
    eps,
):
    X = design_matrix(x)
    sample_size = np.int64(relative_sample_size * len(x))

    best_beta = np.full(2, np.nan)
    best_error = np.inf

    for _ in range(samples):
        beta, error = _lts_single_sample(
            X, y, sample_size, max_iterations, rng, eps=eps
        )
        if error < best_error:
            best_error = error
            best_beta[:] = beta

    return best_beta, best_error
