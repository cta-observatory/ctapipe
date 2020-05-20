from scipy.stats import skew, kurtosis

from ..containers import StatisticsContainer


def descriptive_statistics(
    values, container_class=StatisticsContainer
) -> StatisticsContainer:
    """ compute intensity statistics of an image  """
    return container_class(
        max=values.max(),
        min=values.min(),
        mean=values.mean(),
        std=values.std(),
        skewness=skew(values),
        kurtosis=kurtosis(values),
    )
