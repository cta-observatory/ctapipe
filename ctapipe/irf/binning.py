"""Collection of binning related functionality for the irf tools"""
import astropy.units as u
import numpy as np
from pyirf.binning import create_bins_per_decade

from ..core import Component
from ..core.traits import Float, Integer


def check_bins_in_range(bins, range):
    low = bins >= range.min
    hig = bins <= range.max

    if not all(low & hig):
        raise ValueError(f"Valid range is {range.min} to {range.max}, got {bins}")


class OutputEnergyBinning(Component):
    """Collects energy binning settings."""

    true_energy_min = Float(
        help="Minimum value for True Energy bins in TeV units",
        default_value=0.005,
    ).tag(config=True)

    true_energy_max = Float(
        help="Maximum value for True Energy bins in TeV units",
        default_value=200,
    ).tag(config=True)

    true_energy_n_bins_per_decade = Float(
        help="Number of edges per decade for True Energy bins",
        default_value=10,
    ).tag(config=True)

    reco_energy_min = Float(
        help="Minimum value for Reco Energy bins in TeV units",
        default_value=0.015,
    ).tag(config=True)

    reco_energy_max = Float(
        help="Maximum value for Reco Energy bins in TeV units",
        default_value=200,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Float(
        help="Number of edges per decade for Reco Energy bins",
        default_value=5,
    ).tag(config=True)

    def true_energy_bins(self):
        """
        Creates bins per decade for true MC energy using pyirf function.
        """
        true_energy = create_bins_per_decade(
            self.true_energy_min * u.TeV,
            self.true_energy_max * u.TeV,
            self.true_energy_n_bins_per_decade,
        )
        return true_energy

    def reco_energy_bins(self):
        """
        Creates bins per decade for reconstructed MC energy using pyirf function.
        """
        reco_energy = create_bins_per_decade(
            self.reco_energy_min * u.TeV,
            self.reco_energy_max * u.TeV,
            self.reco_energy_n_bins_per_decade,
        )
        return reco_energy


class FovOffsetBinning(Component):
    """Collects FoV binning settings."""

    fov_offset_min = Float(
        help="Minimum value for FoV Offset bins in degrees",
        default_value=0.0,
    ).tag(config=True)

    fov_offset_max = Float(
        help="Maximum value for FoV offset bins in degrees",
        default_value=5.0,
    ).tag(config=True)

    fov_offset_n_bins = Integer(
        help="Number of edges for FoV offset bins",
        default_value=1,
    ).tag(config=True)

    def fov_offset_bins(self):
        """
        Creates bins for single/multiple FoV offset.
        """
        fov_offset = (
            np.linspace(
                self.fov_offset_min,
                self.fov_offset_max,
                self.fov_offset_n_bins + 1,
            )
            * u.deg
        )
        return fov_offset
