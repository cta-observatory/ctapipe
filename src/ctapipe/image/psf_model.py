"""
Models for the Point Spread Functions of the different telescopes
"""

__all__ = ["PSFModel", "ComaModel"]

from abc import abstractmethod

import numpy as np
from scipy.stats import laplace, laplace_asymmetric

from ctapipe.core.component import non_abstract_children
from ctapipe.core.plugins import detect_and_import_plugins


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

    @classmethod
    def non_abstract_subclasses(cls):
        """
        Get a dict of all non-abstract subclasses of this class.

        This method is using the entry-point plugin system
        to also check for registered plugin implementations.

        Returns
        -------
        subclasses : dict[str, type]
            A dict mapping the name to the class of all found,
            non-abstract  subclasses of this class.
        """
        if hasattr(cls, "plugin_entry_point"):
            detect_and_import_plugins(cls.plugin_entry_point)

        subclasses = {base.__name__: base for base in non_abstract_children(cls)}
        return subclasses

    @abstractmethod
    def pdf(self, *args):
        pass

    @abstractmethod
    def update_location(self, *args):
        pass

    @abstractmethod
    def update_model_parameters(self, *args):
        pass


class ComaModel(PSFModel):
    """
    PSF model, describing pure coma aberrations PSF effect
    """

    def __init__(
        self,
        asymmetry_params=[0.49244797, 9.23573115, 0.15216096],
        radial_scale_params=[0.01409259, 0.02947208, 0.06000271, -0.02969355],
        az_scale_params=[0.24271557, 7.5511501, 0.02037972],
    ):
        """
        PSF model, describing purely the effect of coma aberration on the PSF

        Parameters
        ----------
        asymmetry_params: list
            Parameters describing the dependency of the asymmetry of the psf on the distance to the center of the camera
        radial_scale_params : list
            Parameters describing the dependency of the radial scale on the distance to the center of the camera
        az_scale_params : list
            Parameters describing the dependency of the azimuthal scale scale on the distance to the center of the camera
        """

        self.asymmetry_params = asymmetry_params
        self.radial_scale_params = radial_scale_params
        self.az_scale_params = az_scale_params

        self.update_location(0.0, 0.0)

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
        """
        Calculates the value of the psf at a given location

        Parameters
        ----------
        r : float
            distance to the center of the camera in meters
        f : float
            polar angle in radians

        Returns
        ----------
        psf : float
            value of the PSF at the specified location
        """
        return laplace_asymmetric.pdf(r, *self.radial_pdf_params) * laplace.pdf(
            f, *self.azimuthal_pdf_params
        )

    def update_model_parameters(self, model_params):
        """
        Updates the model parameters for the psf

        Parameters
        ----------
        model_params : dict
            dictionary with the model parameters
            needs to have the keys `asymmetry_params`, `radial_scale_params` and `az_scale_params`
            The values need to be lists of length 3, 4 and 3 respectively
        """
        if not (
            len(model_params["asymmetry_params"]) == 3
            and len(model_params["radial_scale_params"]) == 4
            and len(model_params["az_scale_params"]) == 3
        ):
            raise ValueError(
                "asymmetry_params and az_scale_params needs to have length 3 and radial_scale_params length 4"
            )

        self.asymmetry_params = model_params["asymmetry_params"]
        self.radial_scale_params = model_params["radial_scale_params"]
        self.az_scale_params = model_params["az_scale_params"]
        k = self.k_func(self.radial_pdf_params[1])
        sr = self.sr_func(self.radial_pdf_params[1])
        sf = self.sf_func(self.radial_pdf_params[1])
        self.radial_pdf_params = (k, self.radial_pdf_params[1], sr)
        self.azimuthal_pdf_params = (self.azimuthal_pdf_params[0], sf)

    def update_location(self, r, f):
        """
        Updates the location on the camera focal plane from where the psf is calculated
        Parameters
        ----------
        r : float
            distance to the center of the camera in meters
        f : float
            polar angle in radians
        """
        k = self.k_func(r)
        sr = self.sr_func(r)
        sf = self.sf_func(r)
        self.radial_pdf_params = (k, r, sr)
        self.azimuthal_pdf_params = (f, sf)
