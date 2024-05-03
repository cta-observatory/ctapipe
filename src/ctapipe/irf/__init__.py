"""Top level module for the irf functionality"""
from .binning import FovOffsetBinning, OutputEnergyBinning, check_bins_in_range
from .irfs import (
    Background2dIrf,
    BackgroundIrfBase,
    EffectiveArea2dIrf,
    EffectiveAreaIrfBase,
    EnergyMigration2dIrf,
    EnergyMigrationIrfBase,
    Irf2dBase,
    IrfRecoEnergyBase,
    IrfTrueEnergyBase,
    Psf3dIrf,
    PsfIrfBase,
)
from .optimize import (
    GridOptimizer,
    OptimizationResult,
    OptimizationResultStore,
    ThetaCutsCalculator,
)
from .select import (
    SPECTRA,
    EventPreProcessor,
    EventsLoader,
    Spectra,
)

__all__ = [
    "Irf2dBase",
    "IrfRecoEnergyBase",
    "IrfTrueEnergyBase",
    "PsfIrfBase",
    "BackgroundIrfBase",
    "EnergyMigrationIrfBase",
    "EffectiveAreaIrfBase",
    "Psf3dIrf",
    "Background2dIrf",
    "EnergyMigration2dIrf",
    "EffectiveArea2dIrf",
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
