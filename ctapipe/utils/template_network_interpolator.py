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
        self.interpolator = UnstructuredInterpolator(input_dict, remember_last=True,
                                                     bounds=((-5, 1),(-1.5, 1.5)))
        # function_name="predict")

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
        array = np.stack((energy, impact, xmax), axis=-1)
        points = np.stack((xb, yb), axis=-1)

        interpolated_value = self.interpolator(array, points)
        interpolated_value[interpolated_value<0] = 0
        interpolated_value = interpolated_value

        return interpolated_value
