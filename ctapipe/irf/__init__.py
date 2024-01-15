"""Top level module for the irf functionality"""
from .binning import FovOffsetBinning, OutputEnergyBinning, check_bins_in_range
from .irf_classes import PYIRF_SPECTRA, Spectra
from .irfs import (
    Background2dIrf,
    Background3dIrf,
    EffectiveAreaIrf,
    EnergyMigrationIrf,
    PsfIrf,
)
from .optimize import GridOptimizer, OptimizationResult, OptimizationResultStore
from .select import EventPreProcessor, EventsLoader, ThetaCutsCalculator

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
    "PYIRF_SPECTRA",
    "check_bins_in_range",
]
