from typing import Any

import astropy.units as u
import numpy as np
import tables
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
        help="If bounds_error is False, this flag will specify wether values outside"
        "the available values are filled with nan (False) or extrapolated (True).",
        default_value=False,
    ).tag(config=True)

    def __init__(self, h5file, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(h5file, tables.File):
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

    def _read_pointing_table(self, tel_id):
        pointing_table = read_table(
            self.h5file,
            f"/dl0/monitoring/telescope/pointing/tel_{tel_id:03d}",
        )

        # sort first, so it's not done twice for each interpolator
        pointing_table.sort("time")
        mjd = pointing_table["time"].tai.mjd

        az = pointing_table["azimuth"].quantity.to_value(u.rad)
        # prepare azimuth for interpolation "unwrapping", i.e. turning 359, 1 into 359, 361
        az = np.unwrap(az)
        alt = pointing_table["altitude"].quantity.to_value(u.rad)

        self._az_interpolators[tel_id] = interp1d(mjd, az, **self.interp_options)
        self._alt_interpolators[tel_id] = interp1d(mjd, alt, **self.interp_options)

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
            self._read_pointing_table(tel_id)

        mjd = time.tai.mjd
        az = u.Quantity(self._az_interpolators[tel_id](mjd), u.rad, copy=False)
        alt = u.Quantity(self._alt_interpolators[tel_id](mjd), u.rad, copy=False)
        return alt, az
