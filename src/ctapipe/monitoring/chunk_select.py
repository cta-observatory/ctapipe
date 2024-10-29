import numpy as np

from ctapipe.core import Component, traits


class ChunkFunction(Component):

    """
    Chunk Function for the gain and pedestals
    Interpolates data so that for each time the value from the latest starting
    valid chunk is given or the earliest available still valid chunk for any
    pixels without valid data.

    Parameters
    ----------
    input_table : astropy.table.Table
        Table of calibration values, expected columns
        are always ``start_time`` as ``Start_Time`` column,
        ``end_time`` as ``End_Time`` column
        other columns for the data that is to be selected
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
        self,
        input_table,
        fill_value="extrapolate",
    ):
        input_table.sort("start_time")
        self.start_times = input_table["start_time"]
        self.end_times = input_table["end_time"]
        self.values = input_table["values"]

    def __call__(self, time):
        if time < self.start_times[0]:
            if self.bounds_error:
                raise ValueError("below the interpolation range")

            if self.extrapolate:
                return self.values[0]

            else:
                a = np.empty(self.values[0].shape)
                a[:] = np.nan
                return a

        elif time > self.end_times[-1]:
            if self.bounds_error:
                raise ValueError("above the interpolation range")

            if self.extrapolate:
                return self.values[-1]

            else:
                a = np.empty(self.values[0].shape)
                a[:] = np.nan
                return a

        else:
            i = np.searchsorted(
                self.start_times, time, side="left"
            )  # Latest valid chunk
            j = np.searchsorted(
                self.end_times, time, side="left"
            )  # Earliest valid chunk
            return np.where(
                np.isnan(self.values[i - 1]), self.values[j], self.values[i - 1]
            )  # Give value for latest chunk unless its nan. If nan give earliest chunk value
