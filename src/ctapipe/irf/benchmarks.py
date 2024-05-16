"""Components to generate benchmarks"""
import astropy.units as u
import numpy as np
from astropy.io.fits import BinTableHDU
from astropy.table import QTable
from pyirf.benchmarks import angular_resolution, energy_bias_resolution
from pyirf.binning import create_histogram_table
from pyirf.sensitivity import calculate_sensitivity, estimate_background

from ..core.traits import Bool, Float
from .binning import RecoEnergyBinsBase, TrueEnergyBinsBase
from .spectra import SPECTRA, Spectra


class EnergyBiasResolutionMaker(TrueEnergyBinsBase):
    """
    Calculates the bias and the resolution of the energy prediction in bins of
    true energy.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def make_bias_resolution_hdu(
        self, events: QTable, extname: str = "ENERGY BIAS RESOLUTION"
    ):
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
        bias_resolution = energy_bias_resolution(
            events=events,
            energy_bins=self.true_energy_bins,
            bias_function=np.mean,
            energy_type="true",
        )
        return BinTableHDU(bias_resolution, name=extname)


class AngularResolutionMaker(TrueEnergyBinsBase, RecoEnergyBinsBase):
    """
    Calculates the angular resolution in bins of either true or reconstructed energy.
    """

    # Use reconstructed energy by default for the sake of current pipeline comparisons
    use_true_energy = Bool(
        False,
        help="Use true energy instead of reconstructed energy for energy binning.",
    ).tag(config=True)

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def make_angular_resolution_hdu(
        self, events: QTable, extname: str = "ANGULAR RESOLUTION"
    ):
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
        if self.use_true_energy:
            bins = self.true_energy_bins
            energy_type = "true"
        else:
            bins = self.reco_energy_bins
            energy_type = "reco"

        ang_res = angular_resolution(
            events=events,
            energy_bins=bins,
            energy_type=energy_type,
        )
        return BinTableHDU(ang_res, name=extname)


class SensitivityMaker(RecoEnergyBinsBase):
    """Calculates the point source sensitivity in bins of reconstructed energy."""

    alpha = Float(
        default_value=0.2, help="Ratio between size of the on and the off region."
    ).tag(config=True)

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def make_sensitivity_hdu(
        self,
        signal_events: QTable,
        background_events: QTable,
        theta_cut: QTable,
        fov_offset_min: u.Quantity,
        fov_offset_max: u.Quantity,
        gamma_spectrum: Spectra,
        extname: str = "SENSITIVITY",
    ):
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
        fov_offset_min: astropy.units.Quantity[angle]
            Minimum distance from the fov center for background events to be taken into account.
        fov_offset_max: astropy.units.Quantity[angle]
            Maximum distance from the fov center for background events to be taken into account.
        gamma_spectrum: ctapipe.irf.Spectra
            Spectra by which to scale the relative sensitivity to get the flux sensitivity.
        extname: str
            Name of the BinTableHDU.

        Returns
        -------
        BinTableHDU
        """
        signal_hist = create_histogram_table(
            events=signal_events, bins=self.reco_energy_bins
        )
        bkg_hist = estimate_background(
            events=background_events,
            reco_energy_bins=self.reco_energy_bins,
            theta_cuts=theta_cut,
            alpha=self.alpha,
            fov_offset_min=fov_offset_min,
            fov_offset_max=fov_offset_max,
        )
        sens = calculate_sensitivity(
            signal_hist=signal_hist, background_hist=bkg_hist, alpha=self.alpha
        )
        source_spectrum = SPECTRA[gamma_spectrum]
        sens["flux_sensitivity"] = sens["relative_sensitivity"] * source_spectrum(
            sens["reco_energy_center"]
        )
        return BinTableHDU(sens, name=extname)
