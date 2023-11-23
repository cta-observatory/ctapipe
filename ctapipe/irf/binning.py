import astropy.units as u
import numpy as np
from pyirf.binning import create_bins_per_decade

from ..core import Component
from ..core.traits import Float, Integer


class OutputEnergyBinning(Component):
    """Collects energy binning settings"""

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
        default_value=0.006,
    ).tag(config=True)

    reco_energy_max = Float(
        help="Maximum value for Reco Energy bins in TeV units",
        default_value=190,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Float(
        help="Number of edges per decade for Reco Energy bins",
        default_value=5,
    ).tag(config=True)

    energy_migration_min = Float(
        help="Minimum value of Energy Migration matrix",
        default_value=0.2,
    ).tag(config=True)

    energy_migration_max = Float(
        help="Maximum value of Energy Migration matrix",
        default_value=5,
    ).tag(config=True)

    energy_migration_n_bins = Integer(
        help="Number of bins in log scale for Energy Migration matrix",
        default_value=31,
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

    def energy_migration_bins(self):
        """
        Creates bins for energy migration.
        """
        energy_migration = np.geomspace(
            self.energy_migration_min,
            self.energy_migration_max,
            self.energy_migration_n_bins,
        )
        return energy_migration


class FovOffsetBinning(Component):
    """
    Collects information on generating energy and angular bins for
    generating IRFs as per pyIRF requirements.
    """

    fov_offset_min = Float(
        help="Minimum value for FoV Offset bins in degrees",
        default_value=0.0,
    ).tag(config=True)

    fov_offset_max = Float(
        help="Maximum value for FoV offset bins in degrees",
        default_value=5.0,
    ).tag(config=True)

    fov_offset_n_edges = Integer(
        help="Number of edges for FoV offset bins",
        default_value=2,
    ).tag(config=True)

    def fov_offset_bins(self):
        """
        Creates bins for single/multiple FoV offset.
        """
        fov_offset = (
            np.linspace(
                self.fov_offset_min,
                self.fov_offset_max,
                self.fov_offset_n_edges,
            )
            * u.deg
        )
        return fov_offset


class SourceOffsetBinning(Component):
    """
    Collects information on generating energy and angular bins for
    generating IRFs as per pyIRF requirements.
    """

    source_offset_min = Float(
        help="Minimum value for Source offset for PSF IRF",
        default_value=0,
    ).tag(config=True)

    source_offset_max = Float(
        help="Maximum value for Source offset for PSF IRF",
        default_value=1,
    ).tag(config=True)

    source_offset_n_edges = Integer(
        help="Number of edges for Source offset for PSF IRF",
        default_value=101,
    ).tag(config=True)

    def fov_offset_bins(self):
        """
        Creates bins for single/multiple FoV offset.
        """
        fov_offset = (
            np.linspace(
                self.fov_offset_min,
                self.fov_offset_max,
                self.fov_offset_n_edges,
            )
            * u.deg
        )
        return fov_offset

    def source_offset_bins(self):
        """
        Creates bins for source offset for generating PSF IRF.
        Using the same binning as in pyirf example.
        """

        source_offset = (
            np.linspace(
                self.source_offset_min,
                self.source_offset_max,
                self.source_offset_n_edges,
            )
            * u.deg
        )
        return source_offset
