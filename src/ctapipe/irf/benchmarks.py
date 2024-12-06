"""Components to generate benchmarks"""

from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.io.fits import BinTableHDU, Header
from astropy.table import QTable
from pyirf.benchmarks import angular_resolution, energy_bias_resolution
from pyirf.binning import calculate_bin_indices, create_histogram_table, split_bin_lo_hi
from pyirf.sensitivity import calculate_sensitivity, estimate_background

from ..core.traits import Bool, Float
from .binning import DefaultFoVOffsetBins, DefaultRecoEnergyBins, DefaultTrueEnergyBins
from .spectra import ENERGY_FLUX_UNIT, FLUX_UNIT, SPECTRA, Spectra

__all__ = [
    "EnergyBiasResolutionMakerBase",
    "EnergyBiasResolution2dMaker",
    "AngularResolutionMakerBase",
    "AngularResolution2dMaker",
    "SensitivityMakerBase",
    "Sensitivity2dMaker",
]


def _get_2d_result_table(
    events: QTable, e_bins: u.Quantity, fov_bins: u.Quantity
) -> tuple[QTable, np.ndarray, tuple[int, int]]:
    result = QTable()
    result["ENERG_LO"], result["ENERG_HI"] = split_bin_lo_hi(
        e_bins[np.newaxis, :].to(u.TeV)
    )
    result["THETA_LO"], result["THETA_HI"] = split_bin_lo_hi(
        fov_bins[np.newaxis, :].to(u.deg)
    )
    fov_bin_index, _ = calculate_bin_indices(events["true_source_fov_offset"], fov_bins)
    mat_shape = (len(fov_bins) - 1, len(e_bins) - 1)
    return result, fov_bin_index, mat_shape


class EnergyBiasResolutionMakerBase(DefaultTrueEnergyBins):
    """
    Base class for calculating the bias and resolution of the energy prediciton.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

    @abstractmethod
    def make_bias_resolution_hdu(
        self, events: QTable, extname: str = "ENERGY BIAS RESOLUTION"
    ) -> BinTableHDU:
        """
        Calculate the bias and resolution of the energy prediction.

        Parameters
        ----------
        events: astropy.table.QTable
            Reconstructed events to be used.
        extname: str
            Name of the BinTableHDU.

        Returns
        -------
        BinTableHDU
        """


class EnergyBiasResolution2dMaker(EnergyBiasResolutionMakerBase, DefaultFoVOffsetBins):
    """
    Calculates the bias and the resolution of the energy prediction in bins of
    true energy and fov offset.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def make_bias_resolution_hdu(
        self, events: QTable, extname: str = "ENERGY BIAS RESOLUTION"
    ) -> BinTableHDU:
        result, fov_bin_idx, mat_shape = _get_2d_result_table(
            events=events,
            e_bins=self.true_energy_bins,
            fov_bins=self.fov_offset_bins,
        )
        result["N_EVENTS"] = np.zeros(mat_shape)[np.newaxis, ...]
        result["BIAS"] = np.full(mat_shape, np.nan)[np.newaxis, ...]
        result["RESOLUTION"] = np.full(mat_shape, np.nan)[np.newaxis, ...]

        for i in range(len(self.fov_offset_bins) - 1):
            bias_resolution = energy_bias_resolution(
                events=events[fov_bin_idx == i],
                energy_bins=self.true_energy_bins,
                bias_function=np.mean,
                energy_type="true",
            )
            result["N_EVENTS"][:, i, :] = bias_resolution["n_events"]
            result["BIAS"][:, i, :] = bias_resolution["bias"]
            result["RESOLUTION"][:, i, :] = bias_resolution["resolution"]

        return BinTableHDU(result, name=extname)


class AngularResolutionMakerBase(DefaultTrueEnergyBins, DefaultRecoEnergyBins):
    """
    Base class for calculating the angular resolution.
    """

    # Use reconstructed energy by default for the sake of current pipeline comparisons
    use_true_energy = Bool(
        False,
        help="Use true energy instead of reconstructed energy for energy binning.",
    ).tag(config=True)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

    @abstractmethod
    def make_angular_resolution_hdu(
        self, events: QTable, extname: str = "ANGULAR RESOLUTION"
    ) -> BinTableHDU:
        """
        Calculate the angular resolution.

        Parameters
        ----------
        events: astropy.table.QTable
            Reconstructed events to be used.
        extname: str
            Name of the BinTableHDU.

        Returns
        -------
        BinTableHDU
        """


class AngularResolution2dMaker(AngularResolutionMakerBase, DefaultFoVOffsetBins):
    """
    Calculates the angular resolution in bins of either true or reconstructed energy
    and fov offset.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def make_angular_resolution_hdu(
        self, events: QTable, extname: str = "ANGULAR RESOLUTION"
    ) -> BinTableHDU:
        if self.use_true_energy:
            e_bins = self.true_energy_bins
            energy_type = "true"
        else:
            e_bins = self.reco_energy_bins
            energy_type = "reco"

        result, fov_bin_idx, mat_shape = _get_2d_result_table(
            events=events,
            e_bins=e_bins,
            fov_bins=self.fov_offset_bins,
        )
        result["N_EVENTS"] = np.zeros(mat_shape)[np.newaxis, ...]
        result["ANGULAR_RESOLUTION"] = u.Quantity(
            np.full(mat_shape, np.nan)[np.newaxis, ...], events["theta"].unit
        )

        for i in range(len(self.fov_offset_bins) - 1):
            ang_res = angular_resolution(
                events=events[fov_bin_idx == i],
                energy_bins=e_bins,
                energy_type=energy_type,
            )
            result["N_EVENTS"][:, i, :] = ang_res["n_events"]
            result["ANGULAR_RESOLUTION"][:, i, :] = ang_res["angular_resolution_68"]

        header = Header()
        header["E_TYPE"] = energy_type.upper()
        return BinTableHDU(result, header=header, name=extname)


class SensitivityMakerBase(DefaultRecoEnergyBins):
    """Base class for calculating the point source sensitivity."""

    alpha = Float(
        default_value=0.2,
        help="Size ratio of on region / off region.",
    ).tag(config=True)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

    @abstractmethod
    def make_sensitivity_hdu(
        self,
        signal_events: QTable,
        background_events: QTable,
        theta_cut: QTable,
        gamma_spectrum: Spectra,
        extname: str = "SENSITIVITY",
    ) -> BinTableHDU:
        """
        Calculate the point source sensitivity
        based on ``pyirf.sensitivity.calculate_sensitivity``.

        Parameters
        ----------
        signal_events: astropy.table.QTable
            Reconstructed signal events to be used.
        background_events: astropy.table.QTable
            Reconstructed background events to be used.
        theta_cut: QTable
            Direction cut that was applied on ``signal_events``.
        gamma_spectrum: ctapipe.irf.Spectra
            Spectra by which to scale the relative sensitivity to get the flux sensitivity.
        extname: str
            Name of the BinTableHDU.

        Returns
        -------
        BinTableHDU
        """


class Sensitivity2dMaker(SensitivityMakerBase, DefaultFoVOffsetBins):
    """
    Calculates the point source sensitivity in bins of reconstructed energy
    and fov offset.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def make_sensitivity_hdu(
        self,
        signal_events: QTable,
        background_events: QTable,
        theta_cut: QTable,
        gamma_spectrum: Spectra,
        extname: str = "SENSITIVITY",
    ) -> BinTableHDU:
        source_spectrum = SPECTRA[gamma_spectrum]
        result, fov_bin_idx, mat_shape = _get_2d_result_table(
            events=signal_events,
            e_bins=self.reco_energy_bins,
            fov_bins=self.fov_offset_bins,
        )
        result["N_SIGNAL"] = np.zeros(mat_shape)[np.newaxis, ...]
        result["N_SIGNAL_WEIGHTED"] = np.zeros(mat_shape)[np.newaxis, ...]
        result["N_BACKGROUND"] = np.zeros(mat_shape)[np.newaxis, ...]
        result["N_BACKGROUND_WEIGHTED"] = np.zeros(mat_shape)[np.newaxis, ...]
        result["SIGNIFICANCE"] = np.full(mat_shape, np.nan)[np.newaxis, ...]
        result["RELATIVE_SENSITIVITY"] = np.full(mat_shape, np.nan)[np.newaxis, ...]
        result["FLUX_SENSITIVITY"] = u.Quantity(
            np.full(mat_shape, np.nan)[np.newaxis, ...], FLUX_UNIT
        )
        result["ENERGY_FLUX_SENSITIVITY"] = u.Quantity(
            np.full(mat_shape, np.nan)[np.newaxis, ...], ENERGY_FLUX_UNIT
        )
        for i in range(len(self.fov_offset_bins) - 1):
            signal_hist = create_histogram_table(
                events=signal_events[fov_bin_idx == i], bins=self.reco_energy_bins
            )
            bkg_hist = estimate_background(
                events=background_events,
                reco_energy_bins=self.reco_energy_bins,
                theta_cuts=theta_cut,
                alpha=self.alpha,
                fov_offset_min=self.fov_offset_bins[i],
                fov_offset_max=self.fov_offset_bins[i + 1],
            )
            sens = calculate_sensitivity(
                signal_hist=signal_hist, background_hist=bkg_hist, alpha=self.alpha
            )
            result["N_SIGNAL"][:, i, :] = sens["n_signal"]
            result["N_SIGNAL_WEIGHTED"][:, i, :] = sens["n_signal_weighted"]
            result["N_BACKGROUND"][:, i, :] = sens["n_background"]
            result["N_BACKGROUND_WEIGHTED"][:, i, :] = sens["n_background_weighted"]
            result["SIGNIFICANCE"][:, i, :] = sens["significance"]
            result["RELATIVE_SENSITIVITY"][:, i, :] = sens["relative_sensitivity"]
            result["FLUX_SENSITIVITY"][:, i, :] = (
                sens["relative_sensitivity"]
                * source_spectrum(sens["reco_energy_center"])
            ).to(FLUX_UNIT)
            result["ENERGY_FLUX_SENSITIVITY"][:, i, :] = (
                sens["relative_sensitivity"]
                * source_spectrum(sens["reco_energy_center"])
                * sens["reco_energy_center"] ** 2
            ).to(ENERGY_FLUX_UNIT)

        header = Header()
        header["ALPHA"] = self.alpha
        return BinTableHDU(result, header=header, name=extname)
