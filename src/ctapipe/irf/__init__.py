"""Top level module for the irf functionality"""
from .benchmarks import (
    AngularResolutionMaker,
    EnergyBiasResolutionMaker,
    SensitivityMaker,
)
from .binning import (
    FoVOffsetBinsBase,
    RecoEnergyBinsBase,
    ResultValidRange,
    TrueEnergyBinsBase,
    check_bins_in_range,
    make_bins_per_decade,
)
from .irfs import (
    BackgroundRate2dMaker,
    BackgroundRateMakerBase,
    EffectiveArea2dMaker,
    EffectiveAreaMakerBase,
    EnergyMigration2dMaker,
    EnergyMigrationMakerBase,
    Psf3dMaker,
    PsfMakerBase,
)
from .optimize import (
    CutOptimizerBase,
    GhPercentileCutCalculator,
    OptimizationResult,
    OptimizationResultStore,
    PercentileCuts,
    PointSourceSensitivityOptimizer,
    ThetaPercentileCutCalculator,
)
from .select import EventPreProcessor, EventsLoader
from .spectra import SPECTRA, Spectra

__all__ = [
    "AngularResolutionMaker",
    "EnergyBiasResolutionMaker",
    "SensitivityMaker",
    "TrueEnergyBinsBase",
    "RecoEnergyBinsBase",
    "FoVOffsetBinsBase",
    "PsfMakerBase",
    "BackgroundRateMakerBase",
    "EnergyMigrationMakerBase",
    "EffectiveAreaMakerBase",
    "Psf3dMaker",
    "BackgroundRate2dMaker",
    "EnergyMigration2dMaker",
    "EffectiveArea2dMaker",
    "ResultValidRange",
    "OptimizationResult",
    "OptimizationResultStore",
    "CutOptimizerBase",
    "PointSourceSensitivityOptimizer",
    "PercentileCuts",
    "EventsLoader",
    "EventPreProcessor",
    "Spectra",
    "GhPercentileCutCalculator",
    "ThetaPercentileCutCalculator",
    "SPECTRA",
    "check_bins_in_range",
    "make_bins_per_decade",
]
