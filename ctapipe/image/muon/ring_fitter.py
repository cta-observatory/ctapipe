from ctapipe.core import Component


class RingFitter(Component):
    """This is the base class from which all ring fitters should inherit from"""

    def fit(self, x, y, weight, times=None):
        """overwrite this method with your favourite ring fitting algorithm

        Parameters
        ----------
        x: array
           vector of pixel x-coordinates as astropy quantities
        y: array
           vector of pixel y-coordinates as astropy quantities
        weight: array
           vector of pixel weights
        times: array
           optional vector of pixel DAQ times as astropy quantities

        Returns
        -------
        tuple of centre_x, centre_y, radius, angle, inclination as astropy quantities

        """
        pass
