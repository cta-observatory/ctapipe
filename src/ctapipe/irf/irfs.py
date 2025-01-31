"""Components to generate IRFs"""

from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.io.fits import BinTableHDU
from astropy.table import QTable
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
from pyirf.simulations import SimulatedEventsInfo

from ..core.traits import AstroQuantity, CaselessStrEnum, Float, Integer
from .binning import DefaultFoVOffsetBins, DefaultRecoEnergyBins, DefaultTrueEnergyBins

__all__ = [
    "BackgroundRateMakerBase",
    "BackgroundRate2dMaker",
    "EffectiveAreaMakerBase",
    "EffectiveArea2dMaker",
    "EnergyDispersionMakerBase",
    "EnergyDispersion2dMaker",
    "PSFMakerBase",
    "PSF3DMaker",
]


class PSFMakerBase(DefaultTrueEnergyBins):
    """Base class for calculating the point spread function."""

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

    @abstractmethod
    def __call__(self, events: QTable, extname: str = "PSF") -> BinTableHDU:
        """
        Calculate the psf and create a fits binary table HDU in GADF format.

        Parameters
        ----------
        events: astropy.table.QTable
            Reconstructed events to be used.
        extname: str
            Name for the BinTableHDU.

        Returns
        -------
        BinTableHDU
        """


class BackgroundRateMakerBase(DefaultRecoEnergyBins):
    """Base class for calculating the background rate."""

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

    @abstractmethod
    def __call__(
        self, events: QTable, obs_time: u.Quantity, extname: str = "BACKGROUND"
    ) -> BinTableHDU:
        """
        Calculate the background rate and create a fits binary table HDU
        in GADF format.

        Parameters
        ----------
        events: astropy.table.QTable
            Reconstructed events to be used.
        obs_time: astropy.units.Quantity[time]
            Observation time. This must match with how the individual event
            weights are calculated.
        extname: str
            Name for the BinTableHDU.

        Returns
        -------
        BinTableHDU
        """


class EnergyDispersionMakerBase(DefaultTrueEnergyBins):
    """Base class for calculating the energy dispersion."""

    energy_migration_min = Float(
        help="Minimum value of energy migration ratio",
        default_value=0.2,
    ).tag(config=True)

    energy_migration_max = Float(
        help="Maximum value of energy migration ratio",
        default_value=5,
    ).tag(config=True)

    energy_migration_n_bins = Integer(
        help="Number of bins for energy migration ratio",
        default_value=30,
    ).tag(config=True)

    energy_migration_binning = CaselessStrEnum(
        ["linear", "logarithmic"],
        help=(
            "How energy bins are distributed between energy_migration_min"
            " and energy_migration_max."
        ),
        default_value="logarithmic",
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        bin_func = np.geomspace
        if self.energy_migration_binning == "linear":
            bin_func = np.linspace
        self.migration_bins = bin_func(
            self.energy_migration_min,
            self.energy_migration_max,
            self.energy_migration_n_bins + 1,
        )

    @abstractmethod
    def __call__(
        self,
        events: QTable,
        spatial_selection_applied: bool,
        extname: str = "ENERGY MIGRATION",
    ) -> BinTableHDU:
        """
        Calculate the energy dispersion and create a fits binary table HDU
        in GADF format.

        Parameters
        ----------
        events: astropy.table.QTable
            Reconstructed events to be used.
        spatial_selection_applied: bool
            If a direction cut was applied on ``events``, pass ``True``, else ``False``.
        extname: str
            Name for the BinTableHDU.

        Returns
        -------
        BinTableHDU
        """


class EffectiveAreaMakerBase(DefaultTrueEnergyBins):
    """Base class for calculating the effective area."""

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

    @abstractmethod
    def __call__(
        self,
        events: QTable,
        spatial_selection_applied: bool,
        signal_is_point_like: bool,
        sim_info: SimulatedEventsInfo,
        extname: str = "EFFECTIVE AREA",
    ) -> BinTableHDU:
        """
        Calculate the effective area and create a fits binary table HDU
        in GADF format.

        Parameters
        ----------
        events: astropy.table.QTable
            Reconstructed events to be used.
        spatial_selection_applied: bool
            If a direction cut was applied on ``events``, pass ``True``, else ``False``.
        signal_is_point_like: bool
            If ``events`` were simulated only at a single point in the field of view,
            pass ``True``, else ``False``.
        sim_info: pyirf.simulations.SimulatedEventsInfoa
            The overall statistics of the simulated events.
        extname: str
            Name of the BinTableHDU.

        Returns
        -------
        BinTableHDU
        """


class EffectiveArea2dMaker(EffectiveAreaMakerBase, DefaultFoVOffsetBins):
    """
    Creates a radially symmetric parameterization of the effective area in equidistant
    bins of logarithmic true energy and field of view offset.
    """

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

    def __call__(
        self,
        events: QTable,
        spatial_selection_applied: bool,
        signal_is_point_like: bool,
        sim_info: SimulatedEventsInfo,
        extname: str = "EFFECTIVE AREA",
    ) -> BinTableHDU:
        # For point-like gammas the effective area can only be calculated
        # at one point in the FoV.
        if signal_is_point_like:
            effective_area = effective_area_per_energy(
                selected_events=events,
                simulation_info=sim_info,
                true_energy_bins=self.true_energy_bins,
            )
            # +1 dimension for FOV offset
            effective_area = effective_area[..., np.newaxis]
        else:
            effective_area = effective_area_per_energy_and_fov(
                selected_events=events,
                simulation_info=sim_info,
                true_energy_bins=self.true_energy_bins,
                fov_offset_bins=self.fov_offset_bins,
            )

        return create_aeff2d_hdu(
            effective_area=effective_area,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            point_like=spatial_selection_applied,
            extname=extname,
        )


class EnergyDispersion2dMaker(EnergyDispersionMakerBase, DefaultFoVOffsetBins):
    """
    Creates a radially symmetric parameterization of the energy dispersion in
    equidistant bins of logarithmic true energy and field of view offset.
    """

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

    def __call__(
        self,
        events: QTable,
        spatial_selection_applied: bool,
        extname: str = "ENERGY DISPERSION",
    ) -> BinTableHDU:
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
            point_like=spatial_selection_applied,
            extname=extname,
        )


class BackgroundRate2dMaker(BackgroundRateMakerBase, DefaultFoVOffsetBins):
    """
    Creates a radially symmetric parameterization of the background rate in equidistant
    bins of logarithmic reconstructed energy and field of view offset.
    """

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

    def __call__(
        self, events: QTable, obs_time: u.Quantity, extname: str = "BACKGROUND"
    ) -> BinTableHDU:
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
            extname=extname,
        )


class PSF3DMaker(PSFMakerBase, DefaultFoVOffsetBins):
    """
    Creates a radially symmetric point spread function calculated in equidistant bins
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

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.source_offset_bins = u.Quantity(
            np.linspace(
                self.source_offset_min.to_value(u.deg),
                self.source_offset_max.to_value(u.deg),
                self.source_offset_n_bins + 1,
            ),
            u.deg,
        )

    def __call__(self, events: QTable, extname: str = "PSF") -> BinTableHDU:
        psf = psf_table(
            events=events,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            source_offset_bins=self.source_offset_bins,
        )
        hdu = create_psf_table_hdu(
            psf=psf,
            true_energy_bins=self.true_energy_bins,
            fov_offset_bins=self.fov_offset_bins,
            source_offset_bins=self.source_offset_bins,
            extname=extname,
        )

        return hdu
