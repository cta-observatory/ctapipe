#!/usr/bin/env python3

"""Atmosphere density models and functions to transform between column density
(X in grammage units) and height (meters) units.

Zenith angle is taken into account in the line-of-sight integral to compute the
column density X assuming Earth as a flat plane (the curvature is not taken into
account)

"""

import abc
from dataclasses import dataclass
from functools import partial
from typing import Dict

import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.interpolate import interp1d

__all__ = [
    "AtmosphereDensityProfile",
    "ExponentialAtmosphereDensityProfile",
    "TableAtmosphereDensityProfile",
    "FiveLayerAtmosphereDensityProfile",
]

SUPPORTED_TABLE_VERSIONS = {
    1,
}


class AtmosphereDensityProfile(abc.ABC):
    """
    Base class for models of atmosphere density.
    """

    @abc.abstractmethod
    def __call__(self, height: u.Quantity) -> u.Quantity:
        """
        Returns
        -------
        u.Quantity["g cm-3"]
            the density at height h
        """

    @abc.abstractmethod
    def integral(self, height: u.Quantity) -> u.Quantity:
        r"""Integral of the profile along the height axis, i.e. the *atmospheric
        depth* :math:`X`.

        .. math:: X(h) = \int_{h}^{\infty} \rho(h') dh'

        Returns
        -------
        u.Quantity["g/cm2"]:
            Integral of the density from height h to infinity

        """

    @abc.abstractmethod
    def height_from_overburden(self, overburden: u.Quantity) -> u.Quantity:
        r"""Get the height a.s.l. from the mass overburden in the atmosphere.
            Inverse of the integral function

        ..

        Returns
        -------
        u.Quantity["m"]:
            Height a.s.l. for given overburden

        """

    def line_of_sight_integral(
        self, distance: u.Quantity, zenith_angle=0 * u.deg, output_units=u.g / u.cm**2
    ):
        r"""Line-of-sight integral from the shower distance to infinity, along
        the direction specified by the zenith angle. This is sometimes called
        the *slant depth*. The atmosphere here is assumed to be Cartesian, the
        curvature of the Earth is not taken into account.

        .. math:: X(h, \Psi) = \int_{h}^{\infty} \rho(h' \cos{\Psi}) dh'

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
        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 3))

        fig.suptitle(self.__class__.__name__)
        height = np.linspace(1, 100, 500) * u.km
        density = self(height)
        axis[0].set_xscale("linear")
        axis[0].set_yscale("log")
        axis[0].plot(height, density)
        axis[0].set_xlabel(f"Height / {height.unit.to_string('latex')}")
        axis[0].set_ylabel(f"Density / {density.unit.to_string('latex')}")
        axis[0].grid(True)

        distance = np.linspace(1, 100, 500) * u.km
        for zenith_angle in [0, 40, 50, 70] * u.deg:
            column_density = self.line_of_sight_integral(distance, zenith_angle)
            axis[1].plot(distance, column_density, label=f"$\\Psi$={zenith_angle}")

        axis[1].legend(loc="best")
        axis[1].set_xlabel(f"Distance / {distance.unit.to_string('latex')}")
        axis[1].set_ylabel(f"Column Density / {column_density.unit.to_string('latex')}")
        axis[1].set_yscale("log")
        axis[1].grid(True)

        zenith_angle = np.linspace(0, 80, 20) * u.deg
        for distance in [0, 5, 10, 20] * u.km:
            column_density = self.line_of_sight_integral(distance, zenith_angle)
            axis[2].plot(zenith_angle, column_density, label=f"Height={distance}")

        axis[2].legend(loc="best")
        axis[2].set_xlabel(
            f"Zenith Angle $\\Psi$ / {zenith_angle.unit.to_string('latex')}"
        )
        axis[2].set_ylabel(f"Column Density / {column_density.unit.to_string('latex')}")
        axis[2].set_yscale("log")
        axis[2].grid(True)

        plt.show()
        return fig, axis

    @classmethod
    def from_table(cls, table: Table):
        """return a subclass of AtmosphereDensityProfile from a serialized
        table"""

        if "TAB_TYPE" not in table.meta:
            raise ValueError("expected a TAB_TYPE metadata field")

        version = table.meta.get("TAB_VER", "")
        if version not in SUPPORTED_TABLE_VERSIONS:
            raise ValueError(f"Table version not supported: '{version}'")

        tabtype = table.meta.get("TAB_TYPE")

        if tabtype == "ctapipe.atmosphere.TableAtmosphereDensityProfile":
            return TableAtmosphereDensityProfile(table)
        if tabtype == "ctapipe.atmosphere.FiveLayerAtmosphereDensityProfile":
            return FiveLayerAtmosphereDensityProfile(table)

        raise TypeError(f"Unknown AtmosphereDensityProfile type: '{tabtype}'")


@dataclass
class ExponentialAtmosphereDensityProfile(AtmosphereDensityProfile):
    """
    A simple functional density profile modeled as an exponential.

    The is defined following the form:

    .. math:: \\rho(h) = \\rho_0 e^{-h/h_0}


    .. code-block:: python

        from ctapipe.atmosphere import ExponentialAtmosphereDensityProfile
        density_profile = ExponentialAtmosphereDensityProfile()
        density_profile.peek()


    Attributes
    ----------
    scale_height: u.Quantity["m"]
        scale height (h0)
    scale_density: u.Quantity["g cm-3"]
        scale density (rho0)
    """

    scale_height: u.Quantity = 8 * u.km
    scale_density: u.Quantity = 0.00125 * u.g / (u.cm**3)

    @u.quantity_input(height=u.m)
    def __call__(self, height) -> u.Quantity:
        return self.scale_density * np.exp(-height / self.scale_height)

    @u.quantity_input(height=u.m)
    def integral(
        self,
        height,
    ) -> u.Quantity:
        return (
            self.scale_density * self.scale_height * np.exp(-height / self.scale_height)
        )

    @u.quantity_input(overburden=u.g / (u.cm * u.cm))
    def height_from_overburden(self, overburden) -> u.Quantity:
        return -self.scale_height * np.log(
            overburden / (self.scale_height * self.scale_density)
        )


class TableAtmosphereDensityProfile(AtmosphereDensityProfile):
    """Tabular profile from a table that has both the density and it's integral
    pre-computed.  The table is interpolated to return the density and its integral.

    .. code-block:: python

        from astropy.table import Table
        from astropy import units as u

        from ctapipe.atmosphere import TableAtmosphereDensityProfile

        table = Table(
            dict(
                height=[1,10,20] * u.km,
                density=[0.00099,0.00042, 0.00009] * u.g / u.cm**3
                column_density=[1044.0, 284.0, 57.0] * u.g / u.cm**2
            )
        )

        profile = TableAtmosphereDensityProfile(table=table)
        print(profile(10 * u.km))


    Attributes
    ----------
    table: Table
        Points that define the model

    See Also
    --------
    ctapipe.io.eventsource.EventSource.atmosphere_density_profile:
        load a TableAtmosphereDensityProfile from a supported EventSource
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

        self.table = table[
            (table["height"] >= 0)
            & (table["density"] > 0)
            & (table["column_density"] > 0)
        ]

        # interpolation is done in log-y to minimize spline wobble

        self._density_interp = interp1d(
            self.table["height"].to("km").value,
            np.log10(self.table["density"].to("g cm-3").value),
            kind="cubic",
        )
        self._col_density_interp = interp1d(
            self.table["height"].to("km").value,
            np.log10(self.table["column_density"].to("g cm-2").value),
            kind="cubic",
        )

        self._height_interp = interp1d(
            np.log10(self.table["column_density"].to("g cm-2").value),
            self.table["height"].to("km").value,
            kind="cubic",
        )

        # ensure it can be read back
        self.table.meta["TAB_TYPE"] = "ctapipe.atmosphere.TableAtmosphereDensityProfile"
        self.table.meta["TAB_VER"] = 1

    @u.quantity_input(height=u.m)
    def __call__(self, height) -> u.Quantity:
        return u.Quantity(
            10 ** self._density_interp(height.to_value(u.km)), u.g / u.cm**3
        )

    @u.quantity_input(height=u.m)
    def integral(self, height) -> u.Quantity:
        return u.Quantity(
            10 ** self._col_density_interp(height.to_value(u.km)), u.g / u.cm**2
        )

    @u.quantity_input(overburden=u.g / (u.cm * u.cm))
    def height_from_overburden(self, overburden) -> u.Quantity:
        return u.Quantity(self._height_interp(overburden), u.km)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(meta={self.table.meta}, rows={len(self.table)})"
        )


# Here we define some utility functions needed to build the piece-wise 5-layer
# model.


# pylint: disable=invalid-name,unused-argument
def _exponential(h, a, b, c):
    """exponential atmosphere"""
    return a + b * np.exp(-h / c)


def _d_exponential(h, a, b, c):
    """derivative of exponential atmosphere"""
    return -b / c * np.exp(-h / c)


def _linear(h, a, b, c):
    """linear atmosphere"""
    return a - b * h / c


def _d_linear(h, a, b, c):
    """derivative of linear atmosphere"""
    return -b / c


class FiveLayerAtmosphereDensityProfile(AtmosphereDensityProfile):
    r"""
    CORSIKA 5-layer atmosphere model

    Layers 1-4  are modeled with:

    .. math:: T(h) = a_i + b_i \exp{-h/c_i}

    Layer 5 is modeled with:

    ..math:: T(h) = a_5 - b_5 \frac{h}{c_5}

    References
    ----------
    [corsika-user] D. Heck and T. Pierog, "Extensive Air Shower Simulation with CORSIKA:
        A User’s Guide", 2021, Appendix F
    """

    def __init__(self, table: Table):
        self.table = table

        param_a = self.table["a"].to("g/cm2")
        param_b = self.table["b"].to("g/cm2")
        param_c = self.table["c"].to("km")

        # build list of column density functions and their derivatives:
        self._funcs = [
            partial(f, a=param_a[i], b=param_b[i], c=param_c[i])
            for i, f in enumerate([_exponential] * 4 + [_linear])
        ]
        self._d_funcs = [
            partial(f, a=param_a[i], b=param_b[i], c=param_c[i])
            for i, f in enumerate([_d_exponential] * 4 + [_d_linear])
        ]

    @classmethod
    def from_array(cls, array: np.ndarray, metadata: Dict = None):
        """construct from a 5x5 array as provided by eventio"""

        if metadata is None:
            metadata = {}

        if array.shape != (5, 5):
            raise ValueError("expected ndarray with shape (5,5)")

        table = Table(
            array,
            names=["height", "a", "b", "c", "1/c"],
            units=["cm", "g/cm2", "g/cm2", "cm", "cm-1"],
            meta=metadata,
        )

        table.meta.update(
            dict(
                TAB_VER=1,
                TAB_TYPE="ctapipe.atmosphere.FiveLayerAtmosphereDensityProfile",
            )
        )
        return cls(table)

    @u.quantity_input(height=u.m)
    def __call__(self, height) -> u.Quantity:
        which_func = np.digitize(height, self.table["height"]) - 1
        condlist = [which_func == i for i in range(5)]
        return u.Quantity(
            -1
            * np.piecewise(
                height,
                condlist=condlist,
                funclist=self._d_funcs,
            )
        ).to(u.g / u.cm**3)

    @u.quantity_input(height=u.m)
    def integral(self, height) -> u.Quantity:
        which_func = np.digitize(height, self.table["height"]) - 1
        condlist = [which_func == i for i in range(5)]
        return u.Quantity(
            np.piecewise(
                x=height,
                condlist=condlist,
                funclist=self._funcs,
            )
        ).to(u.g / u.cm**2)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(meta={self.table.meta}, rows={len(self.table)})"
        )
