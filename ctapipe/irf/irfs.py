"""components to generate irfs"""
import astropy.units as u
import numpy as np
from pyirf.binning import create_bins_per_decade
from pyirf.io import (
    create_aeff2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
)
from pyirf.irf import effective_area_per_energy_and_fov, energy_dispersion, psf_table

from ..core import Component
from ..core.traits import Float, Integer
from .binning import check_bins_in_range


class PsfIrf(Component):
    """Collects the functionality for generating PSF IRFs."""

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

    source_offset_min = Float(
        help="Minimum value for Source offset for PSF IRF",
        default_value=0,
    ).tag(config=True)

    source_offset_max = Float(
        help="Maximum value for Source offset for PSF IRF",
        default_value=1,
    ).tag(config=True)

    source_offset_n_bins = Integer(
        help="Number of bins for Source offset for PSF IRF",
        default_value=100,
    ).tag(config=True)

    def __init__(self, parent, valid_offset, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.true_energy_bins = create_bins_per_decade(
            self.true_energy_min * u.TeV,
            self.true_energy_max * u.TeV,
            self.true_energy_n_bins_per_decade,
        )
        self.valid_offset = valid_offset
        self.source_offset_bins = (
            np.linspace(
                self.source_offset_min,
                self.source_offset_max,
                self.source_offset_n_bins + 1,
            )
            * u.deg
        )

    def make_psf_table_hdu(self, signal_events, fov_offset_bins):
        check_bins_in_range(fov_offset_bins, self.valid_offset)
        psf = psf_table(
            events=signal_events,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            source_offset_bins=self.source_offset_bins,
        )
        return create_psf_table_hdu(
            psf,
            self.true_energy_bins,
            self.source_offset_bins,
            fov_offset_bins,
            extname="PSF",
        )


class EnergyMigrationIrf(Component):
    """Collects the functionality for generating Migration Matrix IRFs."""

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

    def __init__(self, parent, **kwargs):
        """
        Creates bins per decade for true MC energy.
        """
        super().__init__(parent=parent, **kwargs)
        self.true_energy_bins = create_bins_per_decade(
            self.true_energy_min * u.TeV,
            self.true_energy_max * u.TeV,
            self.true_energy_n_bins_per_decade,
        )
        self.migration_bins = np.linspace(
            self.energy_migration_min,
            self.energy_migration_max,
            self.energy_migration_n_bins,
        )

    def make_energy_dispersion_hdu(self, signal_events, fov_offset_bins):
        edisp = energy_dispersion(
            signal_events,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            migration_bins=self.migration_bins,
        )
        return create_energy_dispersion_hdu(
            edisp,
            true_energy_bins=self.true_energy_bins,
            migration_bins=self.migration_bins,
            fov_offset_bins=fov_offset_bins,
            extname="ENERGY DISPERSION",
        )


class EffectiveAreaIrf(Component):
    """Collects the functionality for generating Effective Area IRFs."""

    true_energy_min = Float(
        help="Minimum value for True Energy bins in TeV units",
        default_value=0.005,
    ).tag(config=True)

    true_energy_max = Float(
        help="Maximum value for True Energy bins in TeV units",
        default_value=200,
    ).tag(config=True)

    true_energy_n_bins_per_decade = Float(
        help="Number of bins per decade for True Energy bins",
        default_value=10,
    ).tag(config=True)

    def __init__(self, parent, sim_info, **kwargs):
        """
        Creates bins per decade for true MC energy.
        """
        super().__init__(parent=parent, **kwargs)
        self.true_energy_bins = create_bins_per_decade(
            self.true_energy_min * u.TeV,
            self.true_energy_max * u.TeV,
            self.true_energy_n_bins_per_decade,
        )
        self.sim_info = sim_info

    def make_effective_area_hdu(self, signal_events, fov_offset_bins):
        effective_area = effective_area_per_energy_and_fov(
            signal_events,
            self.sim_info,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=fov_offset_bins,
        )
        return create_aeff2d_hdu(
            effective_area[..., np.newaxis],  # +1 dimension for FOV offset
            self.true_energy_bins,
            fov_offset_bins,
            extname="EFFECTIVE AREA",
        )
