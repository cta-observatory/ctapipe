from abc import ABCMeta, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Any

import astropy.units as u
import numpy as np
import tables
from astropy.table import Table
from astropy.time import Time
from scipy.interpolate import interp1d

from ctapipe.core import Component, traits

__all__ = [
    "MonitoringInterpolator",
    "LinearInterpolator",
    "PointingInterpolator",
    "ChunkInterpolator",
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
        This method reads input tables and creates instances of the needed interpolators
        to be added to _interpolators. The first index of _interpolators needs to be
        tel_id, the second needs to be the name of the parameter that is to be interpolated.

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

    def _check_interpolators(self, tel_id: int) -> None:
        if tel_id not in self._interpolators:
            if self.h5file is not None:
                self._read_parameter_table(tel_id)  # might need to be removed
            else:
                raise KeyError(f"No table available for tel_id {tel_id}")

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


class PointingInterpolator(LinearInterpolator):
    """
    Interpolator for pointing and pointing correction data.
    """

    telescope_data_group = "/dl0/monitoring/telescope/pointing"
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

    required_columns = frozenset(["start_time", "end_time"])

    def __init__(self, h5file: None | tables.File = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._interpolators = {}
        self.expected_units = {}
        self.start_time = {}
        self.end_time = {}
        self.values = {}

    def __call__(
        self, tel_id: int, time: Time, columns: str | list[str]
    ) -> float | dict[str, float]:
        """
        Interpolate overlapping chunks of data for a given time, tel_id, and column(s).

        Parameters
        ----------
        tel_id : int
            Telescope id.
        time : astropy.time.Time
            Time for which to interpolate the data.
        columns : str or list of str
            Name(s) of the column(s) to interpolate.

        Returns
        -------
        interpolated : float or dict
            Interpolated data for the specified column(s).
        """

        self._check_interpolators(tel_id)

        if isinstance(columns, str):
            columns = [columns]

        result = {}
        mjd = time.to_value("mjd")
        for column in columns:
            if column not in self._interpolators[tel_id]:
                raise ValueError(
                    f"Column '{column}' not found in interpolators for tel_id {tel_id}"
                )
            result[column] = self._interpolators[tel_id][column](mjd)

        if len(result) == 1:
            return result[columns[0]]
        return result

    def add_table(self, tel_id: int, input_table: Table, columns: list[str]) -> None:
        """
        Add a table to this interpolator for specific columns.

        Parameters
        ----------
        tel_id : int
            Telescope id.
        input_table : astropy.table.Table
            Table of values to be interpolated, expected columns
            are ``start_time`` as ``validity start Time`` column,
            ``end_time`` as ``validity end Time`` and the specified columns
            for the data of the chunks.
        columns : list of str
            Names of the columns to interpolate.
        """

        required_columns = set(deepcopy(self.required_columns))
        required_columns.update(columns)
        self.required_columns = frozenset(required_columns)
        for col in columns:
            self.expected_units[col] = None
        self._check_tables(input_table)

        input_table = input_table.copy()
        input_table.sort("start_time")

        if tel_id not in self._interpolators:
            self._interpolators[tel_id] = {}
            self.values[tel_id] = {}
            self.start_time[tel_id] = {}
            self.end_time[tel_id] = {}

        for column in columns:
            self.values[tel_id][column] = input_table[column]
            self.start_time[tel_id][column] = input_table["start_time"].to_value("mjd")
            self.end_time[tel_id][column] = input_table["end_time"].to_value("mjd")
            self._interpolators[tel_id][column] = partial(
                self._interpolate_chunk, tel_id, column
            )

    def _interpolate_chunk(self, tel_id, column, mjd: float) -> float:
        """
        Interpolates overlapping chunks of data preferring earlier chunks if valid

        Parameters
        ----------
        tel_id : int
            tel_id for which data is to be interpolated
        column : str
            name of the column for which data is to be interpolated
        mjd : float
            Time for which to interpolate the data.
        """

        start_time = self.start_time[tel_id][column]
        end_time = self.end_time[tel_id][column]
        values = self.values[tel_id][column]
        # Find the index of the closest preceding start time
        preceding_index = np.searchsorted(start_time, mjd, side="right") - 1
        if preceding_index < 0:
            return np.nan

        # Check if the time is within the valid range of the chunk
        if start_time[preceding_index] <= mjd <= end_time[preceding_index]:
            value = values[preceding_index]
            if not np.isnan(value):
                return value

        # If the closest preceding chunk has nan, check the next closest chunk
        for i in range(preceding_index - 1, -1, -1):
            if start_time[i] <= mjd <= end_time[i]:
                value = values[i]
                if not np.isnan(value):
                    return value

        return np.nan
