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
from .event_weighter import (
    EventWeighter,
    RadialEventWeighter,
    SimpleEventWeighter,
)
from .irfs import (
    BackgroundRate2dMaker,
    EffectiveArea2dMaker,
    EnergyDispersion2dMaker,
    PSF3DMaker,
)
from .optimize import (
    GhPercentileCutCalculator,
    OptimizationResult,
    PercentileCuts,
    PointSourceSensitivityOptimizer,
    ThetaPercentileCutCalculator,
)
from .spectra import (
    ENERGY_FLUX_UNIT,
    FLUX_UNIT,
    SPECTRA,
    Spectra,
    spectrum_from_name,
    spectrum_from_simulation_config,
)

__all__ = [
    "AngularResolution2dMaker",
    "EnergyBiasResolution2dMaker",
    "Sensitivity2dMaker",
    "PSF3DMaker",
    "BackgroundRate2dMaker",
    "EnergyDispersion2dMaker",
    "EffectiveArea2dMaker",
    "ResultValidRange",
    "OptimizationResult",
    "PointSourceSensitivityOptimizer",
    "PercentileCuts",
    "Spectra",
    "GhPercentileCutCalculator",
    "ThetaPercentileCutCalculator",
    "SPECTRA",
    "ENERGY_FLUX_UNIT",
    "FLUX_UNIT",
    "check_bins_in_range",
    "make_bins_per_decade",
    "EventWeighter",
    "RadialEventWeighter",
    "SimpleEventWeighter",
    "spectrum_from_simulation_config",
    "spectrum_from_name",
]
