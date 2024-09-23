from abc import ABCMeta, abstractmethod
from typing import Any

import astropy.units as u
import numpy as np
import tables
from astropy.time import Time
from scipy.interpolate import interp1d

from ctapipe.core import Component, traits

__all__ = [
    "Interpolator",
    "PointingInterpolator",
]


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
