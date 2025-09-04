from abc import ABCMeta, abstractmethod
from typing import Any

import astropy.units as u
import numpy as np
import tables
from astropy.table import Table
from astropy.time import Time
from scipy.interpolate import interp1d

from ctapipe.core import Component, traits
from ctapipe.core.traits import AstroQuantity
from ctapipe.io.hdf5dataformat import (
    DL0_TEL_POINTING_GROUP,
    DL1_FLATFIELD_IMAGE_GROUP,
    DL1_FLATFIELD_PEAK_TIME_GROUP,
    DL1_SKY_PEDESTAL_IMAGE_GROUP,
)

__all__ = [
    "MonitoringInterpolator",
    "LinearInterpolator",
    "PointingInterpolator",
    "ChunkInterpolator",
    "StatisticsInterpolator",
    "PedestalImageInterpolator",
    "FlatfieldImageInterpolator",
    "FlatfieldPeakTimeInterpolator",
]


class MonitoringInterpolator(Component, metaclass=ABCMeta):
    """
    MonitoringInterpolator parent class.

    Parameters
    ----------
    h5file : None | tables.File
        An open hdf5 file with read access.
    """

    def __init__(self, h5file: None | tables.File = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if h5file is not None and not isinstance(h5file, tables.File):
            raise TypeError("h5file must be a tables.File")
        self.h5file = h5file

    @abstractmethod
    def __call__(self, tel_id: int, time: Time):
        """
        Interpolates monitoring data for a given timestamp

        Parameters
        ----------
        tel_id : int
            Telescope id.
        time : astropy.time.Time
            Time for which to interpolate the monitoring data.

        """
        pass

    @abstractmethod
    def add_table(self, tel_id: int, input_table: Table) -> None:
        """
        Add a table to this interpolator.

        This method reads input tables and creates instances of the needed interpolators.
        The first index of _interpolators needs to be tel_id, the second needs to be
        the name of the parameter that is to be interpolated.

        Parameters
        ----------
        tel_id : int
            Telescope id.
        input_table : astropy.table.Table
            Table of pointing values, expected columns
            are always ``time`` as ``Time`` column and
            other columns for the data that is to be interpolated.
        """
        pass

    def _check_tables(self, input_table: Table) -> None:
        missing = self.required_columns - set(input_table.colnames)
        if len(missing) > 0:
            raise ValueError(f"Table is missing required column(s): {missing}")
        for col in self.expected_units:
            unit = input_table[col].unit
            if unit is None:
                if self.expected_units[col] is not None:
                    raise ValueError(
                        f"{col} must have units compatible with '{self.expected_units[col].name}'"
                    )
            elif not self.expected_units[col].is_equivalent(unit):
                if self.expected_units[col] is None:
                    raise ValueError(f"{col} must have units compatible with 'None'")
                else:
                    raise ValueError(
                        f"{col} must have units compatible with '{self.expected_units[col].name}'"
                    )

    def _read_parameter_table(self, tel_id: int) -> None:
        # prevent circular import between io and monitoring
        from ..io import read_table

        input_table = read_table(
            self.h5file,
            f"{self.telescope_data_group}/tel_{tel_id:03d}",
        )
        self.add_table(tel_id, input_table)


class LinearInterpolator(MonitoringInterpolator):
    """
    LinearInterpolator parent class.

    Parameters
    ----------
    h5file : None | tables.File
        An open hdf5 file with read access.
    """

    bounds_error = traits.Bool(
        default_value=True,
        help="If true, raises an exception when trying to extrapolate out of the given table",
    ).tag(config=True)

    extrapolate = traits.Bool(
        help="If bounds_error is False, this flag will specify whether values outside"
        "the available values are filled with nan (False) or extrapolated (True).",
        default_value=False,
    ).tag(config=True)

    def __init__(self, h5file: None | tables.File = None, **kwargs: Any) -> None:
        super().__init__(h5file, **kwargs)
        self._interpolators = {}
        self.interp_options: dict[str, Any] = dict(assume_sorted=True, copy=False)
        if self.bounds_error:
            self.interp_options["bounds_error"] = True
        elif self.extrapolate:
            self.interp_options["bounds_error"] = False
            self.interp_options["fill_value"] = "extrapolate"
        else:
            self.interp_options["bounds_error"] = False
            self.interp_options["fill_value"] = np.nan

    def _check_interpolators(self, tel_id: int) -> None:
        if tel_id not in self._interpolators:
            if self.h5file is not None:
                self._read_parameter_table(tel_id)  # might need to be removed
            else:
                raise KeyError(f"No table available for tel_id {tel_id}")


class PointingInterpolator(LinearInterpolator):
    """
    Interpolator for pointing and pointing correction data.
    """

    telescope_data_group = DL0_TEL_POINTING_GROUP
    required_columns = frozenset(["time", "azimuth", "altitude"])
    expected_units = {"azimuth": u.rad, "altitude": u.rad}

    def __call__(self, tel_id: int, time: Time) -> tuple[u.Quantity, u.Quantity]:
        """
        Interpolate alt/az for given time and tel_id.

        Parameters
        ----------
        tel_id : int
            Telescope id.
        time : astropy.time.Time
            Time for which to interpolate the pointing.

        Returns
        -------
        altitude : astropy.units.Quantity[deg]
            Interpolated altitude angle.
        azimuth : astropy.units.Quantity[deg]
            Interpolated azimuth angle.
        """

        self._check_interpolators(tel_id)

        mjd = time.tai.mjd
        az = u.Quantity(self._interpolators[tel_id]["az"](mjd), u.rad, copy=False)
        alt = u.Quantity(self._interpolators[tel_id]["alt"](mjd), u.rad, copy=False)
        return alt, az

    def add_table(self, tel_id: int, input_table: Table) -> None:
        """
        Add a table to this interpolator.

        Parameters
        ----------
        tel_id : int
            Telescope id.
        input_table : astropy.table.Table
            Table of pointing values, expected columns
            are ``time`` as ``Time`` column, ``azimuth`` and ``altitude``
            as quantity columns for pointing and pointing correction data.
        """
        self._check_tables(input_table)

        if not isinstance(input_table["time"], Time):
            raise TypeError("'time' column of pointing table must be astropy.time.Time")

        input_table = input_table.copy()
        input_table.sort("time")

        az = input_table["azimuth"].quantity.to_value(u.rad)
        # prepare azimuth for interpolation by "unwrapping": i.e. turning
        # [359, 1] into [359, 361]. This assumes that if we get values like
        # [359, 1] the telescope moved 2 degrees through 0, not 358 degrees
        # the other way around. This should be true for all telescopes given
        # the sampling speed of pointing values and their maximum movement speed.
        # No telescope can turn more than 180° in 2 seconds.
        az = np.unwrap(az)
        alt = input_table["altitude"].quantity.to_value(u.rad)
        mjd = input_table["time"].tai.mjd
        self._interpolators[tel_id] = {}
        self._interpolators[tel_id]["az"] = interp1d(
            mjd, az, kind="linear", **self.interp_options
        )
        self._interpolators[tel_id]["alt"] = interp1d(
            mjd, alt, kind="linear", **self.interp_options
        )


class ChunkInterpolator(MonitoringInterpolator):
    """
    Simple interpolator for overlapping chunks of data.
    """

    timestamp_tolerance = AstroQuantity(
        default_value=u.Quantity(0.1, u.second),
        physical_type=u.physical.time,
        help="Time difference in seconds to consider two timestamps equal.",
    ).tag(config=True)

    def __init__(self, h5file: None | tables.File = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.time_start = {}
        self.time_end = {}
        self.values = {}
        self.columns = list(self.required_columns)  # these will be the data columns
        self.columns.remove("time_start")
        self.columns.remove("time_end")

    def __call__(self, tel_id: int, time: Time) -> float | dict[str, float]:
        """
        Interpolate overlapping chunks of data for a given time, tel_id, and column(s).

        Parameters
        ----------
        tel_id : int
            Telescope id.
        time : astropy.time.Time
            Time for which to interpolate the data.

        Returns
        -------
        interpolated : float or dict
            Interpolated data for the specified column(s).
        """

        if tel_id not in self.values:
            self._read_parameter_table(tel_id)

        result = {}
        for column in self.columns:
            result[column] = self._interpolate_chunk(tel_id, column, time)

        if len(self.columns) == 1:
            return result[self.columns[0]]
        return result

    def add_table(self, tel_id: int, input_table: Table) -> None:
        """
        Add a table to this interpolator for specific columns.

        Parameters
        ----------
        tel_id : int
            Telescope id.
        input_table : astropy.table.Table
            Table of values to be interpolated, expected columns
            are ``time_start`` as ``validity start Time`` column,
            ``time_end`` as ``validity end Time`` and the specified columns
            for the data of the chunks.
        """

        self._check_tables(input_table)

        input_table = input_table.copy()
        input_table.sort("time_start")

        self.values[tel_id] = {}
        self.time_start[tel_id] = input_table["time_start"].to_value("mjd")
        self.time_end[tel_id] = input_table["time_end"].to_value("mjd")

        for column in self.columns:
            self.values[tel_id][column] = input_table[column]

    def _interpolate_chunk(self, tel_id, column, time: Time) -> float | list[float]:
        """
        Interpolates overlapping chunks of data preferring earlier chunks if valid

        Parameters
        ----------
        tel_id : int
            tel_id for which data is to be interpolated
        time : astropy.time.Time
            Time for which to interpolate the data.
        """

        time_start = self.time_start[tel_id]
        time_end = self.time_end[tel_id]
        values = self.values[tel_id][column]
        mjd_times = np.atleast_1d(time.to_value("mjd"))
        # Convert timestamp tolerance to MJD days
        tolerance_mjd = self.timestamp_tolerance.to_value("day")
        # Find the index of the closest preceding start time
        preceding_indices = np.searchsorted(time_start, mjd_times, side="right") - 1

        interpolated_values = []
        for mjd, preceding_index in zip(mjd_times, preceding_indices):
            # Default value is NaN or array of NaNs
            value = (
                np.nan if np.isscalar(values[0]) else np.full_like(values[0], np.nan)
            )
            # Check if the requested time is before the first chunk
            if preceding_index < 0:
                # If the time is before the first chunk and not within tolerance, return NaN
                if (time_start[0] - tolerance_mjd) > mjd:
                    interpolated_values.append(value)
                    continue
                else:
                    # Use the first chunk since it's within tolerance
                    preceding_index = 0

            # Check if the time is within the valid range of the chunk
            if (
                (time_start[preceding_index] - tolerance_mjd)
                <= mjd
                <= (time_end[preceding_index] + tolerance_mjd)
            ):
                value = values[preceding_index]
                # If no NaN values, we can append immediately and continue
                if np.all(~np.isnan(value)):
                    interpolated_values.append(value)
                    continue

            # Fill NaN values from earlier overlapping chunks
            for i in range(preceding_index - 1, -1, -1):
                if (
                    (time_start[i] - tolerance_mjd)
                    <= mjd
                    <= (time_end[i] + tolerance_mjd)
                ):
                    # Only fill NaN values
                    value = np.where(np.isnan(value), values[i], value)
                    # If no NaN values left, we can stop
                    if np.all(~np.isnan(value)):
                        break
            interpolated_values.append(value)
        # If only a single time was provided, return the value directly
        if len(interpolated_values) == 1:
            interpolated_values = interpolated_values[0]
        return interpolated_values


class StatisticsInterpolator(ChunkInterpolator):
    """Interpolator for statistics tables."""

    required_columns = frozenset(["time_start", "time_end", "mean", "median", "std"])
    expected_units = {"mean": None, "median": None, "std": None}


class PedestalImageInterpolator(StatisticsInterpolator):
    """Interpolator for pedestal image tables."""

    telescope_data_group = DL1_SKY_PEDESTAL_IMAGE_GROUP


class FlatfieldImageInterpolator(StatisticsInterpolator):
    """Interpolator for flatfield image tables."""

    telescope_data_group = DL1_FLATFIELD_IMAGE_GROUP


class FlatfieldPeakTimeInterpolator(StatisticsInterpolator):
    """Interpolator for flatfield peak time tables."""

    telescope_data_group = DL1_FLATFIELD_PEAK_TIME_GROUP
