import pickle as pickle
from scipy import interpolate
import numpy as np


class TemplateInterpolator:
    """
    This is a simple class for loading image templates from a pickle file and
    interpolating between them to provide the model for ImPACT style fitting
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
        file = open(filename, 'rb')
        template = pickle.load(file, encoding='latin1')
        grid = pickle.load(file, encoding='latin1')

        self.interpolator = interpolate.LinearNDInterpolator(grid, template)
        self.y_bounds = (-1.5, 1.5)
        self.y_bin_width = (self.y_bounds[1]-self.y_bounds[0])/float(75)

        self.x_bounds = (-5, 1.)
        self.x_bin_width = (self.x_bounds[1] - self.x_bounds[0]) / float(150)

        self.x_bins = np.arange(self.x_bounds[0]+(self.x_bin_width/2),
                                self.x_bounds[1] + (self.x_bin_width / 2), self.x_bin_width)
        self.y_bins = np.arange(self.y_bounds[0]+(self.y_bin_width/2),
                                self.y_bounds[1] + (self.y_bin_width / 2), self.y_bin_width)

        print("Templates Loaded from", filename)

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
        print(image.shape)
        grid_interp = interpolate.RegularGridInterpolator((self.x_bins, self.y_bins), image)

        return grid_interp([pixel_pos_x, pixel_pos_y])

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
        return self.interpolator(params)[0]