#!/usr/bin/env python3

"""Atmosphere density models and functions to transform between column density
(X in grammage units) and height (meters) units.

Zenith angle is taken into account in the line-of-sight integral to compute the
column density X assuming Earth as a flat plane (the curvature is not taken into
account)

"""

import abc
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.interpolate import interp1d

from ..core import Provenance

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
        Returns
        -------
        u.Quantity["g cm-3"]
            the density at height h
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def integral(self, h: u.Quantity, output_units=u.g / u.cm**2) -> u.Quantity:
        """
        Integral of the profile along the height axis

        Returns
        -------
        u.Quantity["g/cm2"]:
            Integral of the density from height h to infinity
        """
        raise NotImplementedError()

    def line_of_sight_integral(
        self, distance: u.Quantity, zenith_angle=0 * u.deg, output_units=u.g / u.cm**2
    ):
        """Line-of-sight integral from the shower distance to infinity, along
        the direction specified by the zenith angle. The atmosphere here is
        assumed to be Cartesian, the curvature of the Earth is not taken into account.

        .. math:: X(h', \\Psi) = \\int_{h'}^{\\infty} \\rho(h \\cos{\\Psi}) dh'

        Parameters
        ----------
        distance: u.Quantity["length"]
           line-of-site distance from observer to point
        zenith_angle: u.Quantity["angle"]
           zenith angle of observation
        output_units: u.Unit
           unit to output (must be convertible to g/cm2)
        """

        return (
            self.integral(distance * np.cos(zenith_angle)) / np.cos(zenith_angle)
        ).to(output_units)

    def peek(self):
        """
        Draw quick plot of profile
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 3))

        fig.suptitle(self.__class__.__name__)
        height = np.geomspace(1, 20, 100) * u.km
        density = self(height)
        ax[0].set_xscale("linear")
        ax[0].set_yscale("log")
        ax[0].plot(height, density)
        ax[0].set_xlabel(f"Height / {height.unit.to_string('latex')}")
        ax[0].set_ylabel(f"Density / {density.unit.to_string('latex')}")
        ax[0].grid(True)

        distance = np.geomspace(1, 20, 100) * u.km
        for zenith_angle in [0, 40, 50, 70] * u.deg:
            column_density = self.line_of_sight_integral(distance, zenith_angle)
            ax[1].plot(distance, column_density, label=f"$\\Psi$={zenith_angle}")

        ax[1].legend(loc="best")
        ax[1].set_xlabel(f"Distance / {height.unit.to_string('latex')}")
        ax[1].set_ylabel(f"Column Density / {density.unit.to_string('latex')}")
        ax[1].grid(True)

        zenith_angle = np.linspace(0, 80, 20) * u.deg
        for distance in [0, 5, 10, 20] * u.km:
            column_density = self.line_of_sight_integral(distance, zenith_angle)
            ax[2].plot(zenith_angle, column_density, label=f"Height={distance}")

        ax[2].legend(loc="best")
        ax[2].set_xlabel(
            f"Zenith Angle $\\Psi$ / {zenith_angle.unit.to_string('latex')}"
        )
        ax[2].set_ylabel(f"Column Density / {density.unit.to_string('latex')}")
        ax[2].grid(True)

        plt.show()


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

    @u.quantity_input
    def __call__(self, h: u.m) -> u.Quantity:
        return self.rho0 * np.exp(-h / self.h0)

    @u.quantity_input
    def integral(
        self,
        h: u.m,
        output_units=u.g / u.cm**2,
    ) -> u.Quantity:
        return self.rho0 * self.h0 * np.exp(-h / self.h0)


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
            if col not in table.colnames:
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

    @u.quantity_input
    def __call__(self, h: u.m) -> u.Quantity:
        return u.Quantity(self._density_interp(h.to_value(u.km)), u.g / u.cm**3)

    @u.quantity_input
    def integral(self, h: u.m) -> u.Quantity:
        return u.Quantity(self._col_density_interp(h.to_value(u.km)), u.g / u.cm**2)

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
        Provenance().add_input_file(
            filename=simtelfile, role="ctapipe.atmosphere.model"
        )

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
