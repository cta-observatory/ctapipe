"""Collection of binning related functionality for the irf tools"""
import astropy.units as u
from pyirf.binning import create_bins_per_decade

from ..core import Component
from ..core.traits import AstroQuantity, Integer


def check_bins_in_range(bins, range, source="result"):
    # `pyirf.binning.create_bins_per_decade` includes the endpoint, if reasonably close.
    # So different choices of `n_bins_per_decade` can lead to mismatches, if the same
    # `*_energy_{min,max}` is chosen.
    low = bins >= range.min * 0.9999999
    hig = bins <= range.max * 1.0000001

    if not all(low & hig):
        raise ValueError(
            f"Valid range for {source} is {range.min} to {range.max}, got {bins}"
        )


class ResultValidRange:
    def __init__(self, bounds_table, prefix):
        self.min = bounds_table[f"{prefix}_min"][0]
        self.max = bounds_table[f"{prefix}_max"][0]


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
