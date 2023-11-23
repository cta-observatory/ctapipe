"""
Define a parent IrfTool class to hold all the options
"""
from enum import Enum

import astropy.units as u
from pyirf.cuts import calculate_percentile_cut
from pyirf.spectral import CRAB_HEGRA, IRFDOC_ELECTRON_SPECTRUM, IRFDOC_PROTON_SPECTRUM

from ..core import Component
from ..core.traits import Float, Integer


class Spectra(Enum):
    CRAB_HEGRA = 1
    IRFDOC_ELECTRON_SPECTRUM = 2
    IRFDOC_PROTON_SPECTRUM = 3


PYIRF_SPECTRA = {
    Spectra.CRAB_HEGRA: CRAB_HEGRA,
    Spectra.IRFDOC_ELECTRON_SPECTRUM: IRFDOC_ELECTRON_SPECTRUM,
    Spectra.IRFDOC_PROTON_SPECTRUM: IRFDOC_PROTON_SPECTRUM,
}


class ThetaCutsCalculator(Component):
    theta_min_angle = Float(
        default_value=-1, help="Smallest angular cut value allowed (-1 means no cut)"
    ).tag(config=True)

    theta_max_angle = Float(
        default_value=0.32, help="Largest angular cut value allowed"
    ).tag(config=True)

    theta_min_counts = Integer(
        default_value=10,
        help="Minimum number of events in a bin to attempt to find a cut value",
    ).tag(config=True)

    theta_fill_value = Float(
        default_value=0.32, help="Angular cut value used for bins with too few events"
    ).tag(config=True)

    theta_smoothing = Float(
        default_value=-1,
        help="When given, the width (in units of bins) of gaussian smoothing applied (-1)",
    ).tag(config=True)

    target_percentile = Float(
        default_value=68,
        help="Percent of events in each energy bin keep after the theta cut",
    ).tag(config=True)

    def calculate_theta_cuts(self, theta, reco_energy, energy_bins):
        theta_min_angle = (
            None if self.theta_min_angle < 0 else self.theta_min_angle * u.deg
        )
        theta_max_angle = (
            None if self.theta_max_angle < 0 else self.theta_max_angle * u.deg
        )
        theta_smoothing = None if self.theta_smoothing < 0 else self.theta_smoothing

        return calculate_percentile_cut(
            theta,
            reco_energy,
            energy_bins,
            min_value=theta_min_angle,
            max_value=theta_max_angle,
            smoothing=theta_smoothing,
            percentile=self.target_percentile,
            fill_value=self.theta_fill_value * u.deg,
            min_events=self.theta_min_counts,
        )
