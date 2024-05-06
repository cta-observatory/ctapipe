"""Components to generate IRFs"""
from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.io.fits import BinTableHDU
from astropy.table import QTable
from pyirf.binning import create_bins_per_decade
from pyirf.io import (
    create_aeff2d_hdu,
    create_background_2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
)
from pyirf.irf import (
    background_2d,
    effective_area_per_energy,
    effective_area_per_energy_and_fov,
    energy_dispersion,
    psf_table,
)

from ..core import Component
from ..core.traits import AstroQuantity, Float, Integer


class IrfTrueEnergyBase(Component):
    """Base class for irf parameterizations binned in true energy."""

    true_energy_min = AstroQuantity(
        help="Minimum value for True Energy bins",
        default_value=u.Quantity(0.015, u.TeV),
        physical_type=u.physical.energy,
    ).tag(config=True)

    true_energy_max = AstroQuantity(
        help="Maximum value for True Energy bins",
        default_value=u.Quantity(150, u.TeV),
        physical_type=u.physical.energy,
    ).tag(config=True)

    true_energy_n_bins_per_decade = Integer(
        help="Number of edges per decade for True Energy bins",
        default_value=10,
    ).tag(config=True)

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.true_energy_bins = create_bins_per_decade(
            self.true_energy_min.to(u.TeV),
            self.true_energy_max.to(u.TeV),
            self.true_energy_n_bins_per_decade,
        )


class IrfRecoEnergyBase(Component):
    """Base class for irf parameterizations binned in reconstructed energy."""

    reco_energy_min = AstroQuantity(
        help="Minimum value for Reco Energy bins",
        default_value=u.Quantity(0.015, u.TeV),
        physical_type=u.physical.energy,
    ).tag(config=True)

    reco_energy_max = AstroQuantity(
        help="Maximum value for Reco Energy bins",
        default_value=u.Quantity(150, u.TeV),
        physical_type=u.physical.energy,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Integer(
        help="Number of edges per decade for Reco Energy bins",
        default_value=10,
    ).tag(config=True)

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.reco_energy_bins = create_bins_per_decade(
            self.reco_energy_min.to(u.TeV),
            self.reco_energy_max.to(u.TeV),
            self.reco_energy_n_bins_per_decade,
        )


class Irf2dBase(Component):
    """Base class for radially symmetric irf parameterizations."""

    fov_offset_min = AstroQuantity(
        help="Minimum value for FoV Offset bins",
        default_value=u.Quantity(0, u.deg),
        physical_type=u.physical.angle,
    ).tag(config=True)

    fov_offset_max = AstroQuantity(
        help="Maximum value for FoV offset bins",
        default_value=u.Quantity(5, u.deg),
        physical_type=u.physical.angle,
    ).tag(config=True)

    fov_offset_n_bins = Integer(
        help="Number of FoV offset bins",
        default_value=1,
    ).tag(config=True)

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.fov_offset_bins = u.Quantity(
            np.linspace(
                self.fov_offset_min.to_value(u.deg),
                self.fov_offset_max.to_value(u.deg),
                self.fov_offset_n_bins + 1,
            ),
            u.deg,
        )


class PsfIrfBase(IrfTrueEnergyBase):
    """Base class for parameterizations of the point spread function."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)

    @abstractmethod
    def make_psf_hdu(self, events: QTable) -> BinTableHDU:
        """
        Calculate the psf and create a fits binary table HDU in GAD format.

        Parameters
        ----------
        events: astropy.table.QTable

        Returns
        -------
        BinTableHDU
        """


class BackgroundIrfBase(IrfRecoEnergyBase):
    """Base class for parameterizations of the background rate."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)

    @abstractmethod
    def make_bkg_hdu(self, events: QTable, obs_time: u.Quantity) -> BinTableHDU:
        """
        Calculate the background rate and create a fits binary table HDU
        in GAD format.

        Parameters
        ----------
        events: astropy.table.QTable
        obs_time: astropy.units.Quantity[time]

        Returns
        -------
        BinTableHDU
        """


class EnergyMigrationIrfBase(IrfTrueEnergyBase):
    """Base class for parameterizations of the energy migration."""

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
        default_value=30,
    ).tag(config=True)

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.migration_bins = np.linspace(
            self.energy_migration_min,
            self.energy_migration_max,
            self.energy_migration_n_bins + 1,
        )

    @abstractmethod
    def make_edisp_hdu(self, events: QTable, point_like: bool) -> BinTableHDU:
        """
        Calculate the energy dispersion and create a fits binary table HDU
        in GAD format.

        Parameters
        ----------
        events: astropy.table.QTable
        point_like: bool

        Returns
        -------
        BinTableHDU
        """


class EffectiveAreaIrfBase(IrfTrueEnergyBase):
    """Base class for parameterizations of the effective area."""

    def __init__(self, parent, sim_info, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.sim_info = sim_info

    @abstractmethod
    def make_aeff_hdu(
        self, events: QTable, point_like: bool, signal_is_point_like: bool
    ) -> BinTableHDU:
        """
        Calculate the effective area and create a fits binary table HDU
        in GAD format.

        Parameters
        ----------
        events: QTable
        point_like: bool
        signal_is_point_like: bool

        Returns
        -------
        BinTableHDU
        """


class EffectiveArea2dIrf(EffectiveAreaIrfBase, Irf2dBase):
    """
    Radially symmetric parameterizations of the effective are in equidistant bins
    of logarithmic true energy and field of view offset.
    """

    def __init__(self, parent, sim_info, **kwargs):
        super().__init__(parent=parent, sim_info=sim_info, **kwargs)

    def make_aeff_hdu(
        self, events: QTable, point_like: bool, signal_is_point_like: bool
    ) -> BinTableHDU:
        # For point-like gammas the effective area can only be calculated
        # at one point in the FoV.
        if signal_is_point_like:
            aeff = effective_area_per_energy(
                selected_events=events,
                simulation_info=self.sim_info,
                true_energy_bins=self.true_energy_bins,
            )
            # +1 dimension for FOV offset
            aeff = aeff[..., np.newaxis]
        else:
            aeff = effective_area_per_energy_and_fov(
                selected_events=events,
                simulation_info=self.sim_info,
                true_energy_bins=self.true_energy_bins,
                fov_offset_bins=self.fov_offset_bins,
            )

        return create_aeff2d_hdu(
            effective_area=aeff,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            point_like=point_like,
            extname="EFFECTIVE AREA",
        )


class EnergyMigration2dIrf(EnergyMigrationIrfBase, Irf2dBase):
    """
    Radially symmetric parameterizations of the energy migration in equidistant
    bins of logarithmic true energy and field of view offset.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def make_edisp_hdu(self, events: QTable, point_like: bool) -> BinTableHDU:
        edisp = energy_dispersion(
            selected_events=events,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            migration_bins=self.migration_bins,
        )
        return create_energy_dispersion_hdu(
            energy_dispersion=edisp,
            true_energy_bins=self.true_energy_bins,
            migration_bins=self.migration_bins,
            fov_offset_bins=self.fov_offset_bins,
            point_like=point_like,
            extname="ENERGY MIGRATION",
        )


class Background2dIrf(BackgroundIrfBase, Irf2dBase):
    """
    Radially symmetric parameterization of the background rate in equidistant
    bins of logarithmic reconstructed energy and field of view offset.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def make_bkg_hdu(self, events: QTable, obs_time: u.Quantity) -> BinTableHDU:
        background_rate = background_2d(
            events=events,
            reco_energy_bins=self.reco_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            t_obs=obs_time,
        )
        return create_background_2d_hdu(
            background_2d=background_rate,
            reco_energy_bins=self.reco_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            extname="BACKGROUND",
        )


class Psf3dIrf(PsfIrfBase, Irf2dBase):
    """
    Radially symmetric point spread function calculated in equidistant bins
    of source offset, logarithmic true energy, and field of view offset.
    """

    source_offset_min = AstroQuantity(
        help="Minimum value for Source offset",
        default_value=u.Quantity(0, u.deg),
        physical_type=u.physical.angle,
    ).tag(config=True)

    source_offset_max = AstroQuantity(
        help="Maximum value for Source offset",
        default_value=u.Quantity(1, u.deg),
        physical_type=u.physical.angle,
    ).tag(config=True)

    source_offset_n_bins = Integer(
        help="Number of bins for Source offset",
        default_value=100,
    ).tag(config=True)

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.source_offset_bins = u.Quantity(
            np.linspace(
                self.source_offset_min.to_value(u.deg),
                self.source_offset_max.to_value(u.deg),
                self.source_offset_n_bins + 1,
            ),
            u.deg,
        )

    def make_psf_hdu(self, events: QTable) -> BinTableHDU:
        psf = psf_table(
            events=events,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            source_offset_bins=self.source_offset_bins,
        )
        return create_psf_table_hdu(
            psf=psf,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            source_offset_bins=self.source_offset_bins,
            extname="PSF",
        )
