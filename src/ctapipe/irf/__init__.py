"""Top level module for the irf functionality"""

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
    OptimizationResultStore,
    PercentileCuts,
    PointSourceSensitivityOptimizer,
    ThetaPercentileCutCalculator,
)
from .select import EventLoader, EventPreProcessor
from .spectra import SPECTRA, Spectra

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
    "OptimizationResultStore",
    "PointSourceSensitivityOptimizer",
    "PercentileCuts",
    "EventLoader",
    "EventPreProcessor",
    "Spectra",
    "GhPercentileCutCalculator",
    "ThetaPercentileCutCalculator",
    "SPECTRA",
    "check_bins_in_range",
    "make_bins_per_decade",
]
