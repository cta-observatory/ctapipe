"""
Models for the Point Spread Functions of the different telescopes
"""

__all__ = ["PSFModel", "ComaModel"]

from abc import abstractmethod

import numpy as np
from scipy.stats import laplace, laplace_asymmetric
from traitlets import List


class PSFModel:
    def __init__(self, **kwargs):
        """
        Base component to describe image distortion due to the optics of the different cameras.
        """

    @classmethod
    def from_name(cls, name, **kwargs):
        """
        Obtain an instance of a subclass via its name

        Parameters
        ----------
        name : str
            Name of the subclass to obtain

        Returns
        -------
        Instance
            Instance of subclass to this class
        """
        requested_subclass = cls.non_abstract_subclasses()[name]
        return requested_subclass(**kwargs)

    @abstractmethod
    def pdf(self, *args):
        pass

    @abstractmethod
    def update_model_parameters(self, *args):
        pass


class ComaModel(PSFModel):
    """
    PSF model, describing pure coma aberrations PSF effect
    """

    asymmetry_params = List(
        default_value=[0.49244797, 9.23573115, 0.15216096],
        help="Parameters describing the dependency of the asymmetry of the psf on the distance to the center of the camera",
    ).tag(config=True)
    radial_scale_params = List(
        default_value=[0.01409259, 0.02947208, 0.06000271, -0.02969355],
        help="Parameters describing the dependency of the radial scale on the distance to the center of the camera",
    ).tag(config=True)
    az_scale_params = List(
        default_value=[0.24271557, 7.5511501, 0.02037972],
        help="Parameters describing the dependency of the azimuthal scale scale on the distance to the center of the camera",
    ).tag(config=True)

    def k_func(self, x):
        return (
            1
            - self.asymmetry_params[0] * np.tanh(self.asymmetry_params[1] * x)
            - self.asymmetry_params[2] * x
        )

    def sr_func(self, x):
        return (
            self.radial_scale_params[0]
            - self.radial_scale_params[1] * x
            + self.radial_scale_params[2] * x**2
            - self.radial_scale_params[3] * x**3
        )

    def sf_func(self, x):
        return self.az_scale_params[0] * np.exp(
            -self.az_scale_params[1] * x
        ) + self.az_scale_params[2] / (self.az_scale_params[2] + x)

    def pdf(self, r, f):
        return laplace_asymmetric.pdf(r, *self.radial_pdf_params) * laplace.pdf(
            f, *self.azimuthal_pdf_params
        )

    def update_model_parameters(self, r, f):
        k = self.k_func(r)
        sr = self.sr_func(r)
        sf = self.sf_func(r)
        self.radial_pdf_params = (k, r, sr)
        self.azimuthal_pdf_params = (f, sf)
