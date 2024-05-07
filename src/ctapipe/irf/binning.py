"""Collection of binning related functionality for the irf tools"""
import astropy.units as u
import numpy as np
from pyirf.binning import create_bins_per_decade

from ..core import Component
from ..core.traits import AstroQuantity, Integer


def check_bins_in_range(bins, range):
    # `pyirf.binning.create_bins_per_decade` includes the endpoint, if reasonably close.
    # So different choices of `n_bins_per_decade` can lead to mismatches, if the same
    # `*_energy_{min,max}` is chosen.
    low = bins >= range.min * 0.9999999
    hig = bins <= range.max * 1.0000001

    if not all(low & hig):
        raise ValueError(f"Valid range is {range.min} to {range.max}, got {bins}")


class OutputEnergyBinning(Component):
    """Collects energy binning settings."""

    true_energy_min = AstroQuantity(
        help="Minimum value for True Energy bins",
        default_value=0.015 * u.TeV,
        physical_type=u.physical.energy,
    ).tag(config=True)

    true_energy_max = AstroQuantity(
        help="Maximum value for True Energy bins",
        default_value=150 * u.TeV,
        physical_type=u.physical.energy,
    ).tag(config=True)

    true_energy_n_bins_per_decade = Integer(
        help="Number of bins per decade for True Energy bins",
        default_value=10,
    ).tag(config=True)

    reco_energy_min = AstroQuantity(
        help="Minimum value for Reco Energy bins",
        default_value=0.015 * u.TeV,
        physical_type=u.physical.energy,
    ).tag(config=True)

    reco_energy_max = AstroQuantity(
        help="Maximum value for Reco Energy bins",
        default_value=150 * u.TeV,
        physical_type=u.physical.energy,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Integer(
        help="Number of bins per decade for Reco Energy bins",
        default_value=5,
    ).tag(config=True)

    def true_energy_bins(self):
        """
        Creates bins per decade for true MC energy using pyirf function.
        """
        true_energy = create_bins_per_decade(
            self.true_energy_min.to(u.TeV),
            self.true_energy_max.to(u.TeV),
            self.true_energy_n_bins_per_decade,
        )
        return true_energy

    def reco_energy_bins(self):
        """
        Creates bins per decade for reconstructed MC energy using pyirf function.
        """
        reco_energy = create_bins_per_decade(
            self.reco_energy_min.to(u.TeV),
            self.reco_energy_max.to(u.TeV),
            self.reco_energy_n_bins_per_decade,
        )
        return reco_energy


class FovOffsetBinning(Component):
    """Collects FoV binning settings."""

    fov_offset_min = AstroQuantity(
        help="Minimum value for FoV Offset bins",
        default_value=0.0 * u.deg,
        physical_type=u.physical.angle,
    ).tag(config=True)

    fov_offset_max = AstroQuantity(
        help="Maximum value for FoV offset bins",
        default_value=5.0 * u.deg,
        physical_type=u.physical.angle,
    ).tag(config=True)

    fov_offset_n_bins = Integer(
        help="Number of bins for FoV offset bins",
        default_value=1,
    ).tag(config=True)

    def fov_offset_bins(self):
        """
        Creates bins for single/multiple FoV offset.
        """
        fov_offset = (
            np.linspace(
                self.fov_offset_min.to_value(u.deg),
                self.fov_offset_max.to_value(u.deg),
                self.fov_offset_n_bins + 1,
            )
            * u.deg
        )
        return fov_offset
