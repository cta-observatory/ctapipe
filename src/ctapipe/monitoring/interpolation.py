from abc import ABCMeta, abstractmethod
from typing import Any

import astropy.units as u
import numpy as np
import tables
from astropy.time import Time
from scipy.interpolate import interp1d

from ctapipe.core import Component, traits

from .astropy_helpers import read_table

class ChunkFunction:

    """
    Chunk Interpolator for the gain and pedestals
    Interpolates data so that for each time the value from the latest starting
    valid chunk is given or the earliest available still valid chunk for any
    pixels without valid data.

    Parameters
    ----------
    values : None | np.array
        Numpy array of the data that is to be interpolated.
        The first dimension needs to be an index over time
    times : None | np.array
        Time values over which data are to be interpolated
        need to be sorted and have same length as first dimension of values
    """

    def __init__(
        self,
        start_times,
        end_times,
        values,
        bounds_error=True,
        fill_value="extrapolate",
        assume_sorted=True,
        copy=False,
    ):
        self.values = values
        self.start_times = start_times
        self.end_times = end_times
        self.bounds_error = bounds_error
        self.fill_value = fill_value

    def __call__(self, point):
        if point < self.start_times[0]:
            if self.bounds_error:
                raise ValueError("below the interpolation range")

            if self.fill_value == "extrapolate":
                return self.values[0]

            else:
                a = np.empty(self.values[0].shape)
                a[:] = np.nan
                return a

        elif point > self.end_times[-1]:
            if self.bounds_error:
                raise ValueError("above the interpolation range")

            if self.fill_value == "extrapolate":
                return self.values[-1]

            else:
                a = np.empty(self.values[0].shape)
                a[:] = np.nan
                return a

        else:
            i = np.searchsorted(
                self.start_times, point, side="left"
            )  # Latest valid chunk
            j = np.searchsorted(
                self.end_times, point, side="left"
            )  # Earliest valid chunk
            return np.where(
                np.isnan(self.values[i - 1]), self.values[j], self.values[i - 1]
            )  # Give value for latest chunk unless its nan. If nan give earliest chunk value


class Interpolator(Component, metaclass=ABCMeta):
    """
    Interpolator parent class.

    Parameters
    ----------
    h5file : None | tables.File
        A open hdf5 file with read access.
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

    telescope_data_group = None
    required_columns = set()
    expected_units = {}

    def __init__(self, h5file=None, **kwargs):
        super().__init__(**kwargs)

        if h5file is not None and not isinstance(h5file, tables.File):
            raise TypeError("h5file must be a tables.File")
        self.h5file = h5file

        self.interp_options: dict[str, Any] = dict(assume_sorted=True, copy=False)
        if self.bounds_error:
            self.interp_options["bounds_error"] = True
        elif self.extrapolate:
            self.interp_options["bounds_error"] = False
            self.interp_options["fill_value"] = "extrapolate"
        else:
            self.interp_options["bounds_error"] = False
            self.interp_options["fill_value"] = np.nan

        self._interpolators = {}

    @abstractmethod
    def add_table(self, tel_id, input_table):
        """
        Add a table to this interpolator
        This method reads input tables and creates instances of the needed interpolators
        to be added to _interpolators. The first index of _interpolators needs to be
        tel_id, the second needs to be the name of the parameter that is to be interpolated

        Parameters
        ----------
        tel_id : int
            Telescope id
        input_table : astropy.table.Table
            Table of pointing values, expected columns
            are always ``time`` as ``Time`` column and
            other columns for the data that is to be interpolated
        """

        pass

    def _check_tables(self, input_table):
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

    def _check_interpolators(self, tel_id):
        if tel_id not in self._interpolators:
            if self.h5file is not None:
                self._read_parameter_table(tel_id)  # might need to be removed
            else:
                raise KeyError(f"No table available for tel_id {tel_id}")

    def _read_parameter_table(self, tel_id):
        # prevent circular import between io and monitoring
        from ..io import read_table

        input_table = read_table(
            self.h5file,
            f"{self.telescope_data_group}/tel_{tel_id:03d}",
        )
        self.add_table(tel_id, input_table)


class PointingInterpolator(Interpolator):
    """
    Interpolator for pointing and pointing correction data
    """

    telescope_data_group = "/dl0/monitoring/telescope/pointing"
    required_columns = frozenset(["time", "azimuth", "altitude"])
    expected_units = {"azimuth": u.rad, "altitude": u.rad}

    def __call__(self, tel_id, time):
        """
        Interpolate alt/az for given time and tel_id.

        Parameters
        ----------
        tel_id : int
            telescope id
        time : astropy.time.Time
            time for which to interpolate the pointing

        Returns
        -------
        altitude : astropy.units.Quantity[deg]
            interpolated altitude angle
        azimuth : astropy.units.Quantity[deg]
            interpolated azimuth angle
        """

        self._check_interpolators(tel_id)

        mjd = time.tai.mjd
        az = u.Quantity(self._interpolators[tel_id]["az"](mjd), u.rad, copy=False)
        alt = u.Quantity(self._interpolators[tel_id]["alt"](mjd), u.rad, copy=False)
        return alt, az

    def add_table(self, tel_id, input_table):
        """
        Add a table to this interpolator

        Parameters
        ----------
        tel_id : int
            Telescope id
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
        self._interpolators[tel_id]["az"] = interp1d(mjd, az, **self.interp_options)
        self._interpolators[tel_id]["alt"] = interp1d(mjd, alt, **self.interp_options)


class FlatFieldInterpolator(Interpolator):
    """
    Interpolator for flatfield data
    """

    telescope_data_group = "dl1/calibration/gain"  # TBD
    required_columns = frozenset(["start_time", "end_time", "gain"])
    expected_units = {"gain": u.one}

    def __call__(self, tel_id, time):
        """
        Interpolate flatfield data for a given time and tel_id.

        Parameters
        ----------
        tel_id : int
            telescope id
        time : astropy.time.Time
            time for which to interpolate the calibration data

        Returns
        -------
        ffield : array [float]
            interpolated flatfield data
        """

        self._check_interpolators(tel_id)

        ffield = self._interpolators[tel_id]["gain"](time)
        return ffield

    def add_table(self, tel_id, input_table):
        """
        Add a table to this interpolator

        Parameters
        ----------
        tel_id : int
            Telescope id
        input_table : astropy.table.Table
            Table of pointing values, expected columns
            are ``time`` as ``Time`` column and "gain"
            for the flatfield data
        """

        self._check_tables(input_table)

        input_table = input_table.copy()
        input_table.sort("start_time")
        start_time = input_table["start_time"]
        end_time = input_table["end_time"]
        gain = input_table["gain"]
        self._interpolators[tel_id] = {}
        self._interpolators[tel_id]["gain"] = ChunkFunction(
            start_time, end_time, gain, **self.interp_options
        )


class PedestalInterpolator(Interpolator):
    """
    Interpolator for Pedestal data
    """

    telescope_data_group = "dl1/calibration/pedestal"  # TBD
    required_columns = frozenset(["start_time", "end_time", "pedestal"])
    expected_units = {"pedestal": u.one}

    def __call__(self, tel_id, time):
        """
        Interpolate pedestal or gain for a given time and tel_id.

        Parameters
        ----------
        tel_id : int
            telescope id
        time : astropy.time.Time
            time for which to interpolate the calibration data

        Returns
        -------
        pedestal : array [float]
            interpolated pedestal values
        """

        self._check_interpolators(tel_id)

        pedestal = self._interpolators[tel_id]["pedestal"](time)
        return pedestal

    def add_table(self, tel_id, input_table):
        """
        Add a table to this interpolator

        Parameters
        ----------
        tel_id : int
            Telescope id
        input_table : astropy.table.Table
            Table of pointing values, expected columns
            are ``time`` as ``Time`` column and "pedestal"
            for the pedestal data
        """

        self._check_tables(input_table)

        input_table = input_table.copy()
        input_table.sort("start_time")
        start_time = input_table["start_time"]
        end_time = input_table["end_time"]
        pedestal = input_table["pedestal"]
        self._interpolators[tel_id] = {}
        self._interpolators[tel_id]["pedestal"] = ChunkFunction(
            start_time, end_time, pedestal, **self.interp_options
        )
