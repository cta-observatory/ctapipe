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


def test_skewness():
    from ctapipe.image.statistics import skewness

    np.random.seed(0)
    data = np.random.normal(5, 2, 1000)
    mean = np.mean(data)
    std = np.std(data)

    assert np.isclose(skewness(data), scipy.stats.skew(data))
    assert np.isclose(skewness(data, mean=mean), scipy.stats.skew(data))
    assert np.isclose(skewness(data, std=std), scipy.stats.skew(data))
    assert np.isclose(skewness(data, mean=mean, std=std), scipy.stats.skew(data))


def test_kurtosis():
    from ctapipe.image.statistics import kurtosis

    np.random.seed(0)
    data = np.random.normal(5, 2, 1000)

    mean = np.mean(data)
    std = np.std(data)

    assert np.isclose(kurtosis(data), scipy.stats.kurtosis(data))
    assert np.isclose(kurtosis(data, mean=mean), scipy.stats.kurtosis(data))
    assert np.isclose(kurtosis(data, std=std), scipy.stats.kurtosis(data))
    assert np.isclose(kurtosis(data, mean=mean, std=std), scipy.stats.kurtosis(data))
    assert np.isclose(
        kurtosis(data, fisher=False), scipy.stats.kurtosis(data, fisher=False)
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
