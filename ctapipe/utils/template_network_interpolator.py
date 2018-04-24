from .unstructured_interpolator import UnstructuredInterpolator
import numpy as np
import pickle
import gzip


class TemplateNetworkInterpolator:
    """
    Class for interpolatating between the the predictions
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

        self.interpolator = UnstructuredInterpolator(input_dict, function_name="predict")

    def __call__(self, energy, impact, xmax, xb, yb):
        """
        Evaluate interpolated templates for a set of shower parameters and pixel positions

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
        array = np.array([energy, impact, xmax])
        points = np.array([xb, yb])
        interpolated_value = self.interpolator(array.T, points.T)
        interpolated_value[interpolated_value<0] = 0

        x_bound = np.logical_and(xb>1, xb<-5)
        y_bound = np.logical_and(yb>1.5, xb<-1.5)

        interpolated_value[x_bound] = 0
        interpolated_value[y_bound] = 0

        return interpolated_value
