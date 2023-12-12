from typing import Any

import astropy.units as u
import numpy as np
import tables
from astropy.time import Time
from scipy.interpolate import interp1d

from ctapipe.core import Component, traits

from .astropy_helpers import read_table


class PointingInterpolator(Component):
    """
    Interpolate pointing from a monitoring table to a given timestamp.

    Monitoring table is expected to be stored at ``/dl0/monitoring/telescope/pointing``
    in the given hdf5 file.
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

        self._alt_interpolators = {}
        self._az_interpolators = {}

    def add_table(self, tel_id, pointing_table):
        """
        Add a table to this interpolator

        Parameters
        ----------
        tel_id : int
            Telescope id
        pointing_table : astropy.table.Table
            Table of pointing values, expected columns
            are ``time`` as ``Time`` column, ``azimuth`` and ``altitude``
            as quantity columns.
        """
        missing = {"time", "azimuth", "altitude"} - set(pointing_table.colnames)
        if len(missing) > 0:
            raise ValueError(f"Table is missing required column(s): {missing}")

        if not isinstance(pointing_table["time"], Time):
            raise TypeError("'time' column of pointing table must be astropy.time.Time")

        for col in ("azimuth", "altitude"):
            unit = pointing_table[col].unit
            if unit is None or not u.rad.is_equivalent(unit):
                raise ValueError(f"{col} must have units compatible with 'rad'")

        # sort first, so it's not done twice for each interpolator
        pointing_table.sort("time")
        # interpolate in mjd TAI. Float64 mjd is precise enough for pointing
        # and TAI is contiguous, so no issues with leap seconds.
        mjd = pointing_table["time"].tai.mjd

        az = pointing_table["azimuth"].quantity.to_value(u.rad)
        # prepare azimuth for interpolation by "unwrapping": i.e. turning
        # [359, 1] into [359, 361]. This assumes that if we get values like
        # [359, 1] the telescope moved 2 degrees through 0, not 358 degrees
        # the other way around. This should be true for all telescopes given
        # the sampling speed of pointing values and their maximum movement speed.
        # No telescope can turn more than 180° in 2 seconds.
        az = np.unwrap(az)
        alt = pointing_table["altitude"].quantity.to_value(u.rad)

        self._az_interpolators[tel_id] = interp1d(mjd, az, **self.interp_options)
        self._alt_interpolators[tel_id] = interp1d(mjd, alt, **self.interp_options)

    def _read_pointing_table(self, tel_id):
        pointing_table = read_table(
            self.h5file,
            f"/dl0/monitoring/telescope/pointing/tel_{tel_id:03d}",
        )
        self.add_table(tel_id, pointing_table)

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
        if tel_id not in self._az_interpolators:
            if self.h5file is not None:
                self._read_pointing_table(tel_id)
            else:
                raise KeyError(f"No pointing table available for tel_id {tel_id}")

        mjd = time.tai.mjd
        az = u.Quantity(self._az_interpolators[tel_id](mjd), u.rad, copy=False)
        alt = u.Quantity(self._alt_interpolators[tel_id](mjd), u.rad, copy=False)
        return alt, az
