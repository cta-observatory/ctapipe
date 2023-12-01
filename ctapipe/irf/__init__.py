"""Top level module for the irf functionality"""
from .binning import FovOffsetBinning, OutputEnergyBinning, SourceOffsetBinning
from .irf_classes import PYIRF_SPECTRA, Spectra
from .optimise import GridOptimizer, OptimisationResult, OptimisationResultSaver
from .select import EventPreProcessor, EventsLoader, ThetaCutsCalculator

__all__ = [
    "OptimisationResult",
    "OptimisationResultSaver",
    "GridOptimizer",
    "OutputEnergyBinning",
    "SourceOffsetBinning",
    "FovOffsetBinning",
    "EventsLoader",
    "EventPreProcessor",
    "Spectra",
    "ThetaCutsCalculator",
    "PYIRF_SPECTRA",
]
