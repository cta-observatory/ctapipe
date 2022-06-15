#!/usr/bin/env python3

"""
Atmosphere density models and functions to transform between column density
(X in grammage units) and height (meters) units.

TODO: include zenith-angle effects
TODO: add simtel models (load from file)
"""

import abc
from dataclasses import dataclass

import numpy as np
from astropy import units as u

__all__ = [
    "AtmosphereDensityProfile",
    "ExponentialAtmosphereDensityProfile",
]


class AtmosphereDensityProfile:
    """
    Base class for models of atmosphere density.
    """

    @abc.abstractmethod
    def __call__(self, h: u.Quantity) -> u.Quantity:
        """
        Return the density at height h
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def integral(self, h: u.Quantity, output_units=u.g / u.cm**2) -> u.Quantity:
        """
        Integral of the profile from height `h` to infinity.
        """
        raise NotImplementedError()


@dataclass
class ExponentialAtmosphereDensityProfile(AtmosphereDensityProfile):
    """
    A simple functional density profile modeled as an exponential.

    The is defined following the form:

    .. math:: \\rho(h) = \\rho_0 e^{-h/h_0}


    Attributes
    ----------
    h0: u.Quantity["m"]
        scale height
    rho0: u.Quantity["g cm-3"]
        scale density
    """

    h0: u.Quantity = 8 * u.km
    rho0: u.Quantity = 0.00125 * u.g / (u.cm**3)

    @u.quantity_input(h=u.m)
    def __call__(self, h: u.Quantity) -> u.Quantity:
        return self.rho0 * np.exp(-h / self.h0)

    @u.quantity_input(h=u.m)
    def integral(self, h: u.Quantity, output_units=u.g / u.cm**2) -> u.Quantity:
        return (self.rho0 * self.h0 * np.exp(-h / self.h0)).to(output_units)
