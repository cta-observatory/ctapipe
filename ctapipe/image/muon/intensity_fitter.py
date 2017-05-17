from ctapipe.core import Component


class IntensityFitter(Component):
    """
    This is the base class from which all muon intensity,
    impact parameter and ring width fitters should inherit from
    """

    def fit(self, x, y, charge, center_x, center_y, radius, times=None):
        """
        overwrite this method with your favourite muon intensity fitting
        algorithm

        Parameters
        ----------
        x: array
           vector of pixel x-coordinates as astropy quantities
        y: array
           vector of pixel y-coordinates as astropy quantities
        charge:
           array of pixel charges as astropy quantities
        center_x:
           previously fitted ring center position x as astropy quantity
        center_y:
           previously fitted ring center position y as astropy quantity
        radius:
           previously fitted ring radius as astropy quantity
        times: array
           optional vector of pixel DAQ times as astropy quantities

        Returns
        -------
        impact_x, impact_y, size, efficiency

        """
        pass
