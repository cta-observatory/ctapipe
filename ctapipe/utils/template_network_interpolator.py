from .unstructured_interpolator import UnstructuredInterpolator
import numpy as np
import pickle
import gzip
import numpy.ma as ma


class BaseTemplate:
    """
    Base class for template interpolators, cheifly associated with code for the special
    interpolation performed on the zenith and azimuth directions. These dimensions are
    treated separately as they should be in priciple generated on a grid, so Delauny
    triangulation based interpolation becomes needlessly inefficient
    """

    def __init__(self):
        self.zeniths = None
        self.azimuths = None
        self.interpolator = None
        self.keys = None
        self.values = None

    def _create_direction_matrix(self, keys, values):

        zeniths = np.sort(np.unique(keys.T[0]))
        azimuths = np.sort(np.unique(keys.T[1]))

        interpolator = np.empty((len(zeniths), len(azimuths)), dtype=object)

        self.interpolator = interpolator
        self.zeniths = zeniths
        self.azimuths = azimuths

        self.keys = keys
        self.values = values

    def _get_bounds(self, zenith, azimuth):

        if len(self.zeniths) == 0:
            zenith_bound = self.zeniths[0], self.zeniths[0]
        else:
            index = np.searchsorted(self.zeniths, zenith, side="left")
            if index != 0:
                index -= 1

            index_upper = index
            if index != len(self.zeniths)-1:
                index_upper += 1

            zenith_bound = (self.zeniths[index], self.zeniths[index_upper])

        if len(self.azimuths) == 0:
            azimuth_bound = self.azimuths[0], self.azimuths[0]
        else:
            index = np.searchsorted(self.azimuths, azimuth)
            if index != 0:
                index -= 1

            index_upper = index
            print(index, self.azimuths, azimuth)

            if index != len(self.azimuths) - 1:
                index_upper += 1
            else:
                index_upper = index-1

            azimuth_bound = (self.azimuths[index], self.azimuths[index_upper])

        return zenith_bound, azimuth_bound


class TemplateNetworkInterpolator(BaseTemplate):
    """
    Class for interpolating between the the predictions
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

        keys = np.array(list(input_dict.keys()))
        values = np.array(list(input_dict.values()))

        # First check if we even have a zen and azimuth entry
        if len(keys[0]) > 3:
            # If we do then for the sake of speed lets
            self._create_direction_matrix(keys, values)
        else:
            # If not we work as before
            self.interpolator = UnstructuredInterpolator(keys, values, remember_last=True,
                                                         bounds=((-5, 1), (-1.5, 1.5)))

    def __call__(self, zenith, azimuth, energy, impact, xmax, xb, yb):
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
        points = ma.dstack((xb, yb))

        interpolated_value = self.interpolator(array, points)
        interpolated_value[interpolated_value<0] = 0
        interpolated_value = interpolated_value

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
        Evaluate expected time gradient for a set of shower parameters and pixel positions

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