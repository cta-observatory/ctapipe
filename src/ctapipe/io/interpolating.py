from typing import Any

import astropy.units as u
import numpy as np
import tables
from astropy.time import Time
from scipy.interpolate import interp1d

from ctapipe.core import Component, traits

from .astropy_helpers import read_table


class StepInterpolator:

    """
    Step function Interpolator for the gain and pedestals

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
        times,
        values,
        bounds_error=True,
        fill_value="extrapolate",
        assume_sorted=True,
        copy=False,
    ):
        self.values = values
        self.times = times
        self.bounds_error = bounds_error
        self.fill_value = fill_value

    def __call__(self, point):
        if point < self.times[0]:
            if self.bounds_error:
                raise ValueError("below the interpolation range")

            if self.fill_value == "extrapolate":
                return self.values[0]

            else:
                a = np.empty(self.values[0].shape)
                a[:] = np.nan
                return a

        else:
            i = np.searchsorted(self.times, point, side="left")
            return self.values[i - 1]


class Interpolator(Component):
    """
    Interpolate pointing and calibration parameters from a monitoring table to a given timestamp.

    Parameters
    ----------
    h5file : None | tables.File
        A open hdf5 file with read access.
    table_location: | str
        location where the monitoring data is expected to be stored in that file
    interpolation_method: | str
        method of interpolation to be used
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

    def __init__(
        self, h5file=None, table_location=None, interpolation_method=None, **kwargs
    ):
        super().__init__(**kwargs)

        if h5file is not None and not isinstance(h5file, tables.File):
            raise TypeError("h5file must be a tables.File")
        self.h5file = h5file

        if table_location is not None and not isinstance(table_location, str):
            raise TypeError("table_location must be a string")
        self.table_location = table_location

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
        self._secondary_interpolators = {}

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
            For pedestal data the quantity column is expected as
            ``pedestal`` and for gain data as ``gain``.
        """

        if "gain" in set(input_table.colnames):
            self.parameter_type = "gain"
            # here i want to detect if the table contains pointing, gain or pedestal information
            # can this be a loop? Should this be a function?
            missing = {"time", "gain"} - set(input_table.colnames)
            if len(missing) > 0:
                raise ValueError(f"Table is missing required column(s): {missing}")

        elif "pedestal" in set(input_table.colnames):
            self.parameter_type = "pedestal"
            missing = {"time", "pedestal"} - set(input_table.colnames)
            if len(missing) > 0:
                raise ValueError(f"Table is missing required column(s): {missing}")

        elif "azimuth" in set(input_table.colnames):
            self.parameter_type = "pointing"
            missing = {"time", "azimuth", "altitude"} - set(input_table.colnames)
            if len(missing) > 0:
                raise ValueError(f"Table is missing required column(s): {missing}")

            for col in ("azimuth", "altitude"):
                unit = input_table[col].unit
                if unit is None or not u.rad.is_equivalent(unit):
                    raise ValueError(f"{col} must have units compatible with 'rad'")

            if not isinstance(input_table["time"], Time):
                raise TypeError(
                    "'time' column of pointing table must be astropy.time.Time"
                )
        # sort first, so it's not done twice for each interpolator
        input_table.sort("time")
        if self.parameter_type == "pointing":
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
            self._interpolators[tel_id] = interp1d(mjd, az, **self.interp_options)
            self._secondary_interpolators[tel_id] = interp1d(
                mjd, alt, **self.interp_options
            )

        elif self.parameter_type == "pedestal":
            time = input_table["time"]

            ped = input_table["pedestal"]

            self._interpolators[tel_id] = StepInterpolator(
                time, ped, **self.interp_options
            )

        elif self.parameter_type == "gain":
            time = input_table["time"]

            gain = input_table["gain"]

            self._interpolators[tel_id] = StepInterpolator(
                time, gain, **self.interp_options
            )

    def _read_parameter_table(self, tel_id, table_location="pointing"):
        if table_location == "pointing":
            table_location = f"/dl0/monitoring/telescope/pointing/tel_{tel_id:03d}"

        input_table = read_table(
            self.h5file,
            table_location,
        )
        self.add_table(tel_id, input_table)

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

        if tel_id not in self._interpolators:
            if self.h5file is not None:
                self._read_parameter_table(tel_id)
            else:
                raise KeyError(f"No table available for tel_id {tel_id}")

        if self.parameter_type == "pointing":
            mjd = time.tai.mjd
            az = u.Quantity(self._interpolators[tel_id](mjd), u.rad, copy=False)
            alt = u.Quantity(
                self._secondary_interpolators[tel_id](mjd), u.rad, copy=False
            )
            return alt, az

        elif self.parameter_type in ("pedestal", "gain"):
            cal = self._interpolators[tel_id](time)
            return cal
