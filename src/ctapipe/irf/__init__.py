"""Top level module for the irf functionality"""
from .binning import FovOffsetBinning, OutputEnergyBinning, check_bins_in_range
from .irfs import (
    Background2dIrf,
    Background3dIrf,
    EffectiveAreaIrf,
    EnergyMigrationIrf,
    PsfIrf,
)
from .optimize import GridOptimizer, OptimizationResult, OptimizationResultStore
from .select import (
    SPECTRA,
    EventPreProcessor,
    EventsLoader,
    Spectra,
    ThetaCutsCalculator,
)

__all__ = [
    "Background2dIrf",
    "Background3dIrf",
    "EffectiveAreaIrf",
    "EnergyMigrationIrf",
    "PsfIrf",
    "OptimizationResult",
    "OptimizationResultStore",
    "GridOptimizer",
    "OutputEnergyBinning",
    "FovOffsetBinning",
    "EventsLoader",
    "EventPreProcessor",
    "Spectra",
    "ThetaCutsCalculator",
    "SPECTRA",
    "check_bins_in_range",
]
