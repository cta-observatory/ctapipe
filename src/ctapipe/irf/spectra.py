"""Definition of spectra to be used to calculate event weights for irf computation"""

import logging
from collections.abc import Callable
from enum import StrEnum

import astropy.units as u
import numpy as np
from astropy.table import Table
from pyirf.simulations import SimulatedEventsInfo
from pyirf.spectral import (
    CRAB_HEGRA,
    IRFDOC_ELECTRON_SPECTRUM,
    IRFDOC_PROTON_SPECTRUM,
    PowerLaw,
)

__all__ = [
    "ENERGY_FLUX_UNIT",
    "FLUX_UNIT",
    "SPECTRA",
    "Spectra",
    "spectrum_from_simulation_config",
]

ENERGY_FLUX_UNIT = (1 * u.erg / u.s / u.cm**2).unit
FLUX_UNIT = (1 / u.erg / u.s / u.cm**2).unit


class Spectra(StrEnum):
    """Spectra for calculating event weights"""

    CRAB_HEGRA = "CRAB_HEGRA"
    IRFDOC_ELECTRON_SPECTRUM = "IRFDOC_ELECTRON_SPECTRUM"
    IRFDOC_PROTON_SPECTRUM = "IRFDOC_PROTON_SPECTRUM"


SPECTRA = {
    Spectra.CRAB_HEGRA: CRAB_HEGRA,
    Spectra.IRFDOC_ELECTRON_SPECTRUM: IRFDOC_ELECTRON_SPECTRUM,
    Spectra.IRFDOC_PROTON_SPECTRUM: IRFDOC_PROTON_SPECTRUM,
}


logger = logging.getLogger(__name__)


def spectrum_from_simulation_config(
    simulation_configuration_table: Table,
    shower_distribution_table: Table | None,
    obs_time: u.Quantity,
    method: str = "powerlaw",
) -> Callable[[u.Quantity], u.Quantity]:
    """
    Return simulated spectrum function from configuration information..

    Note this currently implements only PowerLaw spectra, but in the future will returnq a

    Parameters
    ----------
    simulation_configuration_table: Table
        table as read by TableLoader.read_simulation_configuration()
    shower_distribution_table: Table
        table of simulated shower distribution as read from TableLoader.read_shower_distribution()
    obs_time: u.Quantity['s']
        Observation time in a unit convertible to seconds.
    method: str
        "powerlaw" (for assuming powerlaw distributions), or "histogram" to return an
        interpolation function from the underlying distribution histogram.

    Returns
    -------
    Callable:
       simulated spectrum function.
    """

    if method != "powerlaw":
        return NotImplementedError(f"Method '{method}' is not implemented")

    n_showers: int = 0
    if shower_distribution_table:
        n_showers = shower_distribution_table["n_entries"].sum()
    else:
        logger.warning(
            "Simulation distributions were not found in the input files, "
            "falling back to estimating the number of showers from the "
            "simulation configuration."
        )

    # Some tight consistency checks. Eventually we will support using the
    # arbitrary shower distribution and non-flat spatial distributions.
    # Currently we do not support those, so we raise exceptions here to
    # avoid that we incorrectly compute the effective area, which would have
    # a high scientific impact.
    for itm in [
        "spectral_index",
        "energy_range_min",
        "energy_range_max",
        "max_scatter_range",
        "max_viewcone_radius",
        "min_viewcone_radius",
    ]:
        if len(np.unique(simulation_configuration_table[itm])) > 1:
            raise NotImplementedError(
                f"Unsupported: '{itm}' differs across simulation runs"
            )

    n_showers_config = (
        simulation_configuration_table["n_showers"]
        * simulation_configuration_table["shower_reuse"]
    ).sum()
    if n_showers == 0:
        n_showers = n_showers_config

    assert n_showers_config == n_showers, "Inconsistent number of simulated showers"
    sim_info = SimulatedEventsInfo(
        n_showers=n_showers,
        energy_min=simulation_configuration_table["energy_range_min"].quantity[0],
        energy_max=simulation_configuration_table["energy_range_max"].quantity[0],
        max_impact=simulation_configuration_table["max_scatter_range"].quantity[0],
        spectral_index=simulation_configuration_table["spectral_index"][0],
        viewcone_max=simulation_configuration_table["max_viewcone_radius"].quantity[0],
        viewcone_min=simulation_configuration_table["min_viewcone_radius"].quantity[0],
    )

    return PowerLaw.from_simulation(sim_info, obstime=obs_time)
