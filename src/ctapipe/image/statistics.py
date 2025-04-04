from heapq import nlargest

import numpy as np
from numba import float32, float64, guvectorize, int64, njit

from ..containers import ImageStatisticsContainer

__all__ = [
    "arg_n_largest",
    "arg_n_largest_gu",
    "n_largest",
    "descriptive_statistics",
    "skewness",
    "kurtosis",
]


@njit(cache=True)
def skewness(data, mean=None, std=None):
    """Calculate skewnewss (normalized third central moment)
    with allowing precomputed mean and std.

    With precomputed mean and std, this is ~10x faster than scipy.stats.skew
    for our use case (1D arrays with ~100-1000 elements)

    njit provides ~10% improvement over the non-jitted function.

    Parameters
    ----------
    data: ndarray
        Data for which skewness is calculated
    mean: float or None
        pre-computed mean, if not given, mean is computed
    std: float or None
        pre-computed std, if not given, std is computed

    Returns
    -------
    skewness: float
        computed skewness
    """
    if mean is None:
        mean = np.mean(data)

    if std is None:
        std = np.std(data)

    return np.mean(((data - mean) / std) ** 3)


@njit(cache=True)
def kurtosis(data, mean=None, std=None, fisher=True):
    """Calculate kurtosis (normalized fourth central moment)
    with allowing precomputed mean and std.

    With precomputed mean and std, this is ~10x faster than scipy.stats.skew
    for our use case (1D arrays with ~100-1000 elements)

    njit provides ~10% improvement over the non-jitted function.

    Parameters
    ----------
    data: ndarray
        Data for which skewness is calculated
    mean: float or None
        pre-computed mean, if not given, mean is computed
    std: float or None
        pre-computed std, if not given, std is computed
    fisher: bool
        If True, Fisher’s definition is used (normal ==> 0.0).
        If False, Pearson’s definition is used (normal ==> 3.0).

    Returns
    -------
    kurtosis: float
        kurtosis
    """
    if mean is None:
        mean = np.mean(data)

    if std is None:
        std = np.std(data)

    kurt = np.mean(((data - mean) / std) ** 4)
    if fisher is True:
        kurt -= 3.0
    return kurt


def descriptive_statistics(
    values, container_class=ImageStatisticsContainer
) -> ImageStatisticsContainer:
    """compute intensity statistics of an image"""
    mean = values.mean()
    std = values.std()
    return container_class(
        max=values.max(),
        min=values.min(),
        mean=mean,
        std=std,
        skewness=skewness(values, mean=mean, std=std),
        kurtosis=kurtosis(values, mean=mean, std=std),
    )


@njit
def n_largest(n, array):
    """return the n largest values of an array"""
    return nlargest(n, array)


@guvectorize(
    [
        (int64[:], float64[:], int64[:]),
        (int64[:], float32[:], int64[:]),
    ],
    "(n),(p)->(n)",
    nopython=True,
    cache=True,
)
def arg_n_largest_gu(dummy, array, idx):
    """
    Returns the indices of the len(dummy) largest values of an array.

    Used by the `arg_n_largest` wrapper to return the indices of the n largest
    values of an array.
    """
    values = []
    for i in range(len(array)):
        values.append((array[i], i))
    n = len(dummy)
    largest = n_largest(n, values)
    for i in range(n):
        idx[i] = largest[i][1]


def arg_n_largest(n, array):
    """Return the indices of the n largest values of an array"""
    dummy = np.zeros(n, dtype=np.int64)
    idx = arg_n_largest_gu(dummy, array)
    return idx
