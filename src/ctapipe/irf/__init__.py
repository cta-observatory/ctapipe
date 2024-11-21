"""Top level module for the irf functionality"""

from importlib.util import find_spec

if find_spec("pyirf") is None:
    from ..exceptions import OptionalDependencyMissing

    raise OptionalDependencyMissing("pyirf") from None


from .benchmarks import (
    AngularResolution2dMaker,
    EnergyBiasResolution2dMaker,
    Sensitivity2dMaker,
)
from .binning import (
    ResultValidRange,
    check_bins_in_range,
    make_bins_per_decade,
)
from .irfs import (
    BackgroundRate2dMaker,
    EffectiveArea2dMaker,
    EnergyDispersion2dMaker,
    Psf3dMaker,
)
from .optimize import (
    GhPercentileCutCalculator,
    OptimizationResult,
    PercentileCuts,
    PointSourceSensitivityOptimizer,
    ThetaPercentileCutCalculator,
)
from .preprocessing import EventLoader, EventPreProcessor
from .spectra import ENERGY_FLUX_UNIT, FLUX_UNIT, SPECTRA, Spectra

__all__ = [
    "AngularResolution2dMaker",
    "EnergyBiasResolution2dMaker",
    "Sensitivity2dMaker",
    "Psf3dMaker",
    "BackgroundRate2dMaker",
    "EnergyDispersion2dMaker",
    "EffectiveArea2dMaker",
    "ResultValidRange",
    "OptimizationResult",
    "PointSourceSensitivityOptimizer",
    "PercentileCuts",
    "EventLoader",
    "EventPreProcessor",
    "Spectra",
    "GhPercentileCutCalculator",
    "ThetaPercentileCutCalculator",
    "SPECTRA",
    "ENERGY_FLUX_UNIT",
    "FLUX_UNIT",
    "check_bins_in_range",
    "make_bins_per_decade",
]
