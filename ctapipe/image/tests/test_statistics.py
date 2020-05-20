import numpy as np
from scipy.stats import skew, kurtosis


def test_statistics():
    from ctapipe.image import descriptive_statistics

    np.random.seed(0)
    data = np.random.normal(5, 2, 1000)

    stats = descriptive_statistics(data)

    assert np.isclose(stats.mean, np.mean(data))
    assert np.isclose(stats.std, np.std(data))
    assert np.isclose(stats.skewness, skew(data))
    assert np.isclose(stats.kurtosis, kurtosis(data))
