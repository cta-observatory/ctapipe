"""
Pedestal Calculation Functions
"""

import numpy as np


def calc_pedestals_from_traces(traces, start_sample, end_sample):
    """A very simple algorithm to calculates pedestals and pedestal
    variances from camera traces by integrating the samples over a
    fixed window for all pixels.  This assumes that the data are
    sample-mode (e.g. cameras that return time traces for each pixel).

    Parameters
    ----------

    traces: array of shape (n_pixels, n_samples)
        time-sampled camera data in a 2D array pixel x sample
    start_sample: int
        index of starting sample over which to integrate
    end_sample: int
        index of ending sample over which to integrate

    Returns
    -------

    two arrays of length n_pix (the first dimension of the input trace
    array). The first array contains the pedestal values, and the
    second is the pedestal variances over the sample window.

    """
    traces = np.asanyarray(traces)  # ensure this is an ndarray
    peds = traces[:, start_sample:end_sample].mean(axis=1)
    pedvars = traces[:, start_sample:end_sample].var(axis=1)
    return peds, pedvars


# PUT OTHER PEDESTAL CALCULATION FUNCTIONS HERE:
# ----------------------------------------------
