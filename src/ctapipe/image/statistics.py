import astropy.units as u
import numpy as np
from astropy.stats import circmean, circstd
from astropy.table import Table
from numba import njit

from ..containers import (
    ArrayEventContainer,
    BaseStatisticsContainer,
    StatisticsContainer,
)
from ..core import Component
from ..core.traits import List, Tuple, Unicode
from ..vectorization import max_ufunc, min_ufunc, weighted_mean_ufunc

__all__ = ["descriptive_statistics", "skewness", "kurtosis", "FeatureAggregator"]


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
    values, container_class=StatisticsContainer
) -> StatisticsContainer:
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


class FeatureAggregator(Component):
    """Array-event-wise aggregation of image parameters."""

    image_parameters = List(
        Tuple(Unicode(), Unicode()),
        default_value=[],
        help=(
            "List of 2-Tuples of Strings: ('prefix', 'feature'). "
            "The image parameter to be aggregated is 'prefix_feature'."
        ),
    ).tag(config=True)

    def __call__(self, event: ArrayEventContainer) -> None:
        """Fill event container with aggregated image parameters."""
        for prefix, feature in self.image_parameters:
            values = []
            unit = None
            for tel_id in event.dl1.tel.keys():
                value = event.dl1.tel[tel_id].parameters[prefix][feature]
                if isinstance(value, u.Quantity):
                    if not unit:
                        unit = value.unit
                    value = value.to_value(unit)

                valid = value >= 0 if prefix == "morphology" else ~np.isnan(value)
                if valid:
                    values.append(value)

            if len(values) > 0:
                if feature.endswith(("psi", "phi")):
                    mean = circmean(
                        u.Quantity(values, unit, copy=False).to_value(u.rad)
                    )
                    std = circstd(u.Quantity(values, unit, copy=False).to_value(u.rad))
                else:
                    mean = np.mean(values)
                    std = np.std(values)

                # Use the same dtype for all columns, independent of the dtype
                # of the aggregated image parameter, since `_mean` and `_std`
                # requiere floats anyway.
                max = np.float64(np.max(values))
                min = np.float64(np.min(values))
            else:
                max = np.nan
                min = np.nan
                mean = np.nan
                std = np.nan

            if unit:
                max = u.Quantity(max, unit, copy=False)
                min = u.Quantity(min, unit, copy=False)
                if feature.endswith(("psi", "phi")):
                    mean = u.Quantity(mean, u.rad, copy=False).to(unit)
                    std = u.Quantity(std, u.rad, copy=False).to(unit)
                else:
                    mean = u.Quantity(mean, unit, copy=False)
                    std = u.Quantity(std, unit, copy=False)

            event.dl1.aggregate[prefix + "_" + feature] = BaseStatisticsContainer(
                max=max, min=min, mean=mean, std=std, prefix=prefix + "_" + feature
            )

    def aggregate_table(self, mono_parameters: Table) -> dict[str, Table]:
        """
        Construct table containing aggregated image parameters
        from table of telescope events.
        """
        agg_tables = {}
        for prefix, feature in self.image_parameters:
            col = prefix + "_" + feature
            unit = mono_parameters[col].quantity.unit
            if prefix == "morphology":
                valid = mono_parameters[col] >= 0
            else:
                valid = ~np.isnan(mono_parameters[col])

            valid_parameters = mono_parameters[valid]
            array_events, indices, multiplicity = np.unique(
                mono_parameters["obs_id", "event_id"],
                return_inverse=True,
                return_counts=True,
            )
            agg_table = Table(array_events)
            for colname in ("obs_id", "event_id"):
                agg_table[colname].description = mono_parameters[colname].description

            n_array_events = len(array_events)
            if len(valid_parameters) > 0:
                mono_column = valid_parameters[col]
                means = weighted_mean_ufunc(
                    mono_column,
                    np.array([1]),
                    n_array_events,
                    indices[valid],
                )
                # FIXME: This has the same problem of strange NaNs as
                # e.g. "energy_uncert" generated by the StereoMeanCombiner!
                # Output is also incorrect, but I don't understand why...
                vars = weighted_mean_ufunc(
                    (mono_column - np.repeat(means, multiplicity)[valid]) ** 2,
                    np.array([1]),
                    n_array_events,
                    indices[valid],
                )
                max = max_ufunc(
                    mono_column,
                    n_array_events,
                    indices[valid],
                )
                min = min_ufunc(
                    mono_column,
                    n_array_events,
                    indices[valid],
                )
            else:
                means = np.full(n_array_events, np.nan)
                vars = np.full(n_array_events, np.nan)
                max = np.full(n_array_events, np.nan)
                min = np.full(n_array_events, np.nan)

            agg_table[col + "_max"] = u.Quantity(max, unit, copy=False)
            agg_table[col + "_min"] = u.Quantity(min, unit, copy=False)
            agg_table[col + "_mean"] = u.Quantity(means, unit, copy=False)
            agg_table[col + "_std"] = u.Quantity(np.sqrt(vars), unit, copy=False)

            agg_tables[col] = agg_table

        return agg_tables
