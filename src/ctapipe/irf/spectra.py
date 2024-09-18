"""Definition of spectra to be used to calculate event weights for irf computation"""

from enum import Enum

import astropy.units as u
from pyirf.spectral import CRAB_HEGRA, IRFDOC_ELECTRON_SPECTRUM, IRFDOC_PROTON_SPECTRUM

FLUX_UNIT = (1 * u.erg / u.s / u.cm**2).unit


class Spectra(Enum):
    """Spectra for calculating event weights"""

    CRAB_HEGRA = 1
    IRFDOC_ELECTRON_SPECTRUM = 2
    IRFDOC_PROTON_SPECTRUM = 3


SPECTRA = {
    Spectra.CRAB_HEGRA: CRAB_HEGRA,
    Spectra.IRFDOC_ELECTRON_SPECTRUM: IRFDOC_ELECTRON_SPECTRUM,
    Spectra.IRFDOC_PROTON_SPECTRUM: IRFDOC_PROTON_SPECTRUM,
}
