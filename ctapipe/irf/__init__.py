"""Top level module for the irf functionality"""
from .binning import FovOffsetBinning, OutputEnergyBinning, check_bins_in_range
from .irf_classes import PYIRF_SPECTRA, Spectra
from .irfs import EffectiveAreaIrf, EnergyMigrationIrf, PsfIrf
from .optimise import GridOptimizer, OptimisationResult, OptimisationResultStore
from .select import EventPreProcessor, EventsLoader, ThetaCutsCalculator

__all__ = [
    "EnergyMigrationIrf",
    "PsfIrf",
    "EffectiveAreaIrf",
    "OptimisationResult",
    "OptimisationResultStore",
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
