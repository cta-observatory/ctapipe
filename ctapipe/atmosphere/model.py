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
from astropy.table import Table
from scipy.interpolate import interp1d

__all__ = [
    "AtmosphereDensityProfile",
    "ExponentialAtmosphereDensityProfile",
    "TableAtmosphereDensityProfile",
    "read_simtel_profile",
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


class TableAtmosphereDensityProfile(AtmosphereDensityProfile):
    """Tabular profile from a table that has both the density and it's integral
    pre-computed.  The table is interpolated to return the density and its integral.

    Attributes
    ----------
    table: Table
        Points that define the model
    """

    def __init__(self, table: Table):
        """
        Parameters
        ----------
        table: Table
            Table with columns `height`, `density`, and `column_density`
        """

        for col in ["height", "density", "column_density"]:
            if not col in table.colnames:
                raise ValueError(f"Missing expected column: {col} in table")

        self.table = table
        self._density_interp = interp1d(
            table["height"].to("km"), table["density"].to("g cm-3")
        )
        self._col_density_interp = interp1d(
            table["height"].to("km"), table["column_density"].to("g cm-2")
        )

    @classmethod
    def from_simtel(
        cls, simtel_filename: str, profile_number: int = 0
    ) -> AtmosphereDensityProfile:
        """Construct a TableAtmosphereDensityProfile from a simtel file
        containing a set of atmosphere profiles.

        Parameters
        ----------
        simtel_filename: str
            filename of a SimTelArray data file
        profile_number: int
            index of profile in the file if there are more than one
        """
        return cls(table=read_simtel_profile(simtel_filename)[profile_number])

    def __call__(self, h: u.Quantity) -> u.Quantity:
        return self._density_interp(h.to_value(u.km)) * u.g / u.cm**3

    def integral(self, h: u.Quantity) -> u.Quantity:
        return self._col_density_interp(h.to_value(u.km)) * u.g / u.cm**2

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(meta={self.table.meta}, rows={len(self.table)})"
        )


def read_simtel_profile(simtelfile: str) -> Table:
    """
    Read an atmosphere profile from a SimTelArray file as an astropy Table

    Returns
    -------
    Table:
        table with columns `height`, `density`, and `column_density`
        along with associated metadata
    """
    import eventio

    tables = []
    with eventio.SimTelFile(simtelfile) as simtel:
        for atmo in simtel.atmospheric_profiles:
            table = Table(
                dict(
                    height=atmo["altitude_km"] * u.km,
                    density=atmo["rho"] * u.g / u.cm**3,
                    column_density=atmo["thickness"] * u.g / u.cm**2,
                ),
                meta=dict(
                    obs_level=atmo["obslevel"] * u.cm,
                    atmo_id=atmo["id"],
                    atmo_name=atmo["name"],
                    htoa=atmo["htoa"],  # what is this?,
                ),
            )
            tables.append(table)
    return tables
