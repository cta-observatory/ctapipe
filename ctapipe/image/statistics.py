import numpy as np

from ..containers import StatisticsContainer


def skewness(data, mean=None, std=None):
    '''Calculate skewnewss (normalized third central moment)
    with allowing precomputed mean and std.

    With precomputed mean and std, this is ~10x faster than scipy.stats.skew
    for our use case (1D arrays with ~100-1000 elements)

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
    '''
    mean = mean or np.mean(data)
    std = std or np.std(data)
    return np.average(((data - mean) / std)**3)


def kurtosis(data, mean=None, std=None, fisher=True):
    '''Calculate kurtosis (normalized fourth central moment)
    with allowing precomputed mean and std.

    With precomputed mean and std, this is ~10x faster than scipy.stats.skew
    for our use case (1D arrays with ~100-1000 elements)

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
    '''
    mean = mean or np.mean(data)
    std = std or np.std(data)
    kurt = np.average(((data - mean) / std)**4)
    if fisher is True:
        kurt -= 3.0
    return kurt


def descriptive_statistics(
    values, container_class=StatisticsContainer
) -> StatisticsContainer:
    """ compute intensity statistics of an image  """
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
