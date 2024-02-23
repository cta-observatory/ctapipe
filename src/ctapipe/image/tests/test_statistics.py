import astropy.units as u
import numpy as np
import scipy.stats

from ctapipe.containers import (
    ArrayEventContainer,
    HillasParametersContainer,
    ImageParametersContainer,
    MorphologyContainer,
)


def test_statistics():
    from ctapipe.image import descriptive_statistics

    rng = np.random.default_rng(0)
    data = rng.normal(5, 2, 1000)

    stats = descriptive_statistics(data)

    assert np.isclose(stats.mean, np.mean(data))
    assert np.isclose(stats.std, np.std(data))
    assert np.isclose(stats.skewness, scipy.stats.skew(data))
    assert np.isclose(stats.kurtosis, scipy.stats.kurtosis(data))


def test_skewness():
    from ctapipe.image.statistics import skewness

    rng = np.random.default_rng(0)
    data = rng.normal(5, 2, 1000)
    mean = np.mean(data)
    std = np.std(data)

    assert np.isclose(skewness(data), scipy.stats.skew(data))
    assert np.isclose(skewness(data, mean=mean), scipy.stats.skew(data))
    assert np.isclose(skewness(data, std=std), scipy.stats.skew(data))
    assert np.isclose(skewness(data, mean=mean, std=std), scipy.stats.skew(data))


def test_kurtosis():
    from ctapipe.image.statistics import kurtosis

    rng = np.random.default_rng(0)
    data = rng.normal(5, 2, 1000)

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

    rng = np.random.default_rng(0)
    data = rng.normal(5, 2, 1000)

    stats = descriptive_statistics(data)
    assert isinstance(stats, StatisticsContainer)

    stats = descriptive_statistics(data, container_class=PeakTimeStatisticsContainer)
    assert isinstance(stats, PeakTimeStatisticsContainer)


def test_feature_aggregator():
    from ctapipe.image import FeatureAggregator

    event = ArrayEventContainer()
    for tel_id, length, n_islands in zip((2, 7, 11), (0.3, 0.5, 0.4), (2, 1, 3)):
        event.dl1.tel[tel_id].parameters = ImageParametersContainer(
            hillas=HillasParametersContainer(length=length * u.deg),
            morphology=MorphologyContainer(n_islands=n_islands),
        )

    features = [
        ("hillas", "length"),
        ("morphology", "n_islands"),
    ]
    aggregate_featuers = FeatureAggregator(image_parameters=features)
    aggregate_featuers(event)
    assert event.dl1.aggregate["hillas_length"].max == 0.5 * u.deg
    assert event.dl1.aggregate["hillas_length"].min == 0.3 * u.deg
    assert u.isclose(event.dl1.aggregate["hillas_length"].mean, 0.4 * u.deg)
    assert u.isclose(event.dl1.aggregate["hillas_length"].std, 0.081649658 * u.deg)
    assert event.dl1.aggregate["morphology_n_islands"].max == 3
    assert event.dl1.aggregate["morphology_n_islands"].min == 1
    assert np.isclose(event.dl1.aggregate["morphology_n_islands"].mean, 2)
    assert np.isclose(event.dl1.aggregate["morphology_n_islands"].std, 0.81649658)
