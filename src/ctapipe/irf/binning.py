"""Collection of binning related functionality for the irf tools"""

import logging

import astropy.units as u
import numpy as np

from ..core import Component
from ..core.traits import AstroQuantity, Integer

logger = logging.getLogger(__name__)


def check_bins_in_range(bins, range, source="result", raise_error=True):
    low = bins >= range.min
    hig = bins <= range.max

    if not all(low & hig):
        with np.printoptions(edgeitems=2, threshold=6, precision=4):
            bins = np.array2string(bins)
            min_val = np.array2string(range.min)
            max_val = np.array2string(range.max)
            if raise_error:
                raise ValueError(
                    f"Valid range for {source} is {min_val} to {max_val}, got {bins}"
                )
            else:
                logger.warning(
                    f"Valid range for {source} is {min_val} to {max_val}, got {bins}",
                )


@u.quantity_input(e_min=u.TeV, e_max=u.TeV)
def make_bins_per_decade(e_min, e_max, n_bins_per_decade=5):
    """
    Create energy bins with at least ``bins_per_decade`` bins per decade.
    The number of bins is calculated as
    ``n_bins = ceil((log10(e_max) - log10(e_min)) * n_bins_per_decade)``.

    Parameters
    ----------
    e_min: u.Quantity[energy]
        Minimum energy, inclusive
    e_max: u.Quantity[energy]
        Maximum energy, inclusive
    n_bins_per_decade: int
        Minimum number of bins per decade

    Returns
    -------
    bins: u.Quantity[energy]
        The created bin array, will have units of ``e_min``
    """
    unit = e_min.unit
    log_lower = np.log10(e_min.to_value(unit))
    log_upper = np.log10(e_max.to_value(unit))

    n_bins = int(np.ceil((log_upper - log_lower) * n_bins_per_decade))

    return u.Quantity(np.logspace(log_lower, log_upper, n_bins), unit, copy=False)


class ResultValidRange:
    def __init__(self, bounds_table, prefix):
        self.min = bounds_table[f"{prefix}_min"][0]
        self.max = bounds_table[f"{prefix}_max"][0]


class TrueEnergyBinsBase(Component):
    """Base class for creating irfs or benchmarks binned in true energy."""

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
        help="Number of bins per decade for True Energy bins",
        default_value=10,
    ).tag(config=True)

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.true_energy_bins = make_bins_per_decade(
            self.true_energy_min.to(u.TeV),
            self.true_energy_max.to(u.TeV),
            self.true_energy_n_bins_per_decade,
        )


class RecoEnergyBinsBase(Component):
    """Base class for creating irfs or benchmarks binned in reconstructed energy."""

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
        help="Number of bins per decade for Reco Energy bins",
        default_value=5,
    ).tag(config=True)

    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.reco_energy_bins = make_bins_per_decade(
            self.reco_energy_min.to(u.TeV),
            self.reco_energy_max.to(u.TeV),
            self.reco_energy_n_bins_per_decade,
        )


class FoVOffsetBinsBase(Component):
    """Base class for creating radially symmetric irfs or benchmarks."""

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
