"""Top level module for the irf functionality"""
from .binning import FovOffsetBinning, OutputEnergyBinning, check_bins_in_range
from .irfs import (
    BackgroundRate2dMaker,
    BackgroundRateMakerBase,
    EffectiveArea2dMaker,
    EffectiveAreaMakerBase,
    EnergyMigration2dMaker,
    EnergyMigrationMakerBase,
    IrfMaker2dBase,
    IrfMakerRecoEnergyBase,
    IrfMakerTrueEnergyBase,
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
    "IrfMaker2dBase",
    "IrfMakerRecoEnergyBase",
    "IrfMakerTrueEnergyBase",
    "PsfMakerBase",
    "BackgroundRateMakerBase",
    "EnergyMigrationMakerBase",
    "EffectiveAreaMakerBase",
    "Psf3dMaker",
    "BackgroundRate2dMaker",
    "EnergyMigration2dMaker",
    "EffectiveArea2dMaker",
    "OptimizationResult",
    "OptimizationResultStore",
    "CutOptimizerBase",
    "PointSourceSensitivityOptimizer",
    "PercentileCuts",
    "OutputEnergyBinning",
    "FovOffsetBinning",
    "EventsLoader",
    "EventPreProcessor",
    "Spectra",
    "GhPercentileCutCalculator",
    "ThetaPercentileCutCalculator",
    "SPECTRA",
    "check_bins_in_range",
]
