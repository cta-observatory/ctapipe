"""
TODO:
- Deal better with interpolation outside of the range bounded by keys (currently returns 0)
-
"""

import pickle as pickle
from scipy import interpolate
import numpy as np
import gzip
from ctapipe.utils import fitshistogram


class TableInterpolator:
    """
    This is a simple class for loading image templates from a pickle file and
    interpolating between them to provide the model for ImPACT style fitting. This interpolation scheme
    only supports interpolation between tables of equal size and binning.
    """

    def __init__(self, filename):
        """
        Initialisation of class to load templates from a file and create the interpolation
        objects

        Parameters
        ----------
        filename: string
            Location of Template file
        """
        file = gzip.open(filename, 'rb')
        tables = pickle.load(file, encoding='latin1')
        template_hist = np.asarray(list(tables.values()))
        grid = np.asarray(list(tables.keys()))

        template = list()
        x_bins = template_hist[0].bin_lower_edges()[0]
        y_bins = template_hist[0].bin_lower_edges()[1]

        for t in range(template_hist.shape[0]):
            template.append(template_hist[t].hist)

        self.interpolator = interpolate.LinearNDInterpolator(grid, template[:], fill_value=0)
        self.nearest_interpolator = interpolate.NearestNDInterpolator(grid, template)

        self.grid_interp = interpolate.RegularGridInterpolator((x_bins, y_bins),
                                                               np.zeros([x_bins.shape[0], y_bins.shape[0]]),
                                                               method="linear", bounds_error=False, fill_value=0)
        file.close()

        print("Tables Loaded from", filename)

    def interpolate(self, params, pixel_pos_x, pixel_pos_y):
        """

        Parameters
        ----------
        params: ndarray
            numpy array of interpolation parameters
            currently [energy, impact distance, xmax]
        pixel_pos_x: ndarray
            pixel position in degrees
        pixel_pos_y: ndarray
            pixel position in degrees

        Returns
        -------
        ndarray of expected intensity for all pixel positions given

        """

        image = self.interpolated_image(params)
        self.grid_interp.values = image
        points = np.array([pixel_pos_x, pixel_pos_y])
        return self.grid_interp(points.T)

    def interpolated_image(self, params):
        """
        Function for creating a ful interpolated image template from the interpolation library

        Parameters
        ----------
        params: ndarray
            numpy array of interpolation parameters
            currently [energy, impact distance, xmax]

        Returns
        -------
        ndarray of a single image template

        """
        image = self.interpolator(params)[0]
        if np.isnan(image).all():
            print("Found a NaN", params)
            image = self.nearest_interpolator(params)[0]

        return image
