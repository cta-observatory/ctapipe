import numpy as np
import scipy.stats


def test_statistics():
    from ctapipe.image import descriptive_statistics

    np.random.seed(0)
    data = np.random.normal(5, 2, 1000)

    stats = descriptive_statistics(data)

    assert np.isclose(stats.mean, np.mean(data))
    assert np.isclose(stats.std, np.std(data))
    assert np.isclose(stats.skewness, scipy.stats.skew(data))
    assert np.isclose(stats.kurtosis, scipy.stats.kurtosis(data))


def test_kurtosis():
    from ctapipe.image.statistics import kurtosis

    np.random.seed(0)
    data = np.random.normal(5, 2, 1000)

    assert np.isclose(kurtosis(data), scipy.stats.kurtosis(data))
    assert np.isclose(
        kurtosis(data, fisher=False),
        scipy.stats.kurtosis(data, fisher=False),
    )


def test_return_type():
    from ctapipe.containers import PeakTimeStatisticsContainer, StatisticsContainer
    from ctapipe.image import descriptive_statistics

    np.random.seed(0)
    data = np.random.normal(5, 2, 1000)

    stats = descriptive_statistics(data)
    assert isinstance(stats, StatisticsContainer)

    stats = descriptive_statistics(data, container_class=PeakTimeStatisticsContainer)
    assert isinstance(stats, PeakTimeStatisticsContainer)
