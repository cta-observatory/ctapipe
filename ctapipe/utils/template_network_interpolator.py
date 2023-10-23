import gzip
import pickle

import numpy as np
import numpy.ma as ma

from .unstructured_interpolator import UnstructuredInterpolator


class TemplateNetworkInterpolator:
    """
    Class for interpolating ImPACT templates.
    """

    def __init__(self, template_file):
        """

        Parameters
        ----------
        template_file: str
            Location of pickle file containing ImPACT NN templates
        """

        file_list = gzip.open(template_file)
        input_dict = pickle.load(file_list)
        self.interpolator = UnstructuredInterpolator(
            input_dict, remember_last=True, bounds=((-5, 1), (-1.5, 1.5))
        )

    def reset(self):
        """
        Reset method to delete some saved results from the previous event
        """
        self.interpolator.reset()

    def __call__(self, energy, impact, xmax, xb, yb):
        """
        Evaluate interpolated templates at given parameters.

        Parameters
        ----------
        energy: array-like
            Energy of interpolated template
        impact: array-like
            Impact distance of interpolated template
        xmax: array-like
            Depth of maximum of interpolated templates
        xb: array-like
            Pixel X position at which to evaluate template
        yb: array-like
            Pixel X position at which to evaluate template

        Returns
        -------
        ndarray: Pixel amplitude expectation values
        """
        array = np.stack((energy, impact, xmax), axis=-1)
        points = ma.dstack((xb, yb))

        interpolated_value = self.interpolator(array, points)
        interpolated_value[interpolated_value < 0] = 0

        return interpolated_value


class TimeGradientInterpolator:
    """
    Class for interpolating between the time gradient predictions
    """

    def __init__(self, template_file):
        """

        Parameters
        ----------
        template_file: str
            Location of pickle file containing ImPACT NN templates
        """

        file_list = gzip.open(template_file)
        input_dict = pickle.load(file_list)
        self.interpolator = UnstructuredInterpolator(input_dict, remember_last=False)

    def __call__(self, energy, impact, xmax):
        """
        Evaluate expected time gradient at given parameters.

        Parameters
        ----------
        energy: array-like
            Energy of interpolated template
        impact: array-like
            Impact distance of interpolated template
        xmax: array-like
            Depth of maximum of interpolated templates

        Returns
        -------
        ndarray: Time Gradient expectation and RMS values
        """
        array = np.stack((energy, impact, xmax), axis=-1)

        interpolated_value = self.interpolator(array)

        return interpolated_value
