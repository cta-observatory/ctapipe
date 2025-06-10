"""module containing optimization related functions and classes"""

import operator
from abc import abstractmethod

import astropy.units as u
import numpy as np
from astropy.table import QTable
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut

from ...core import Component
from ...core.traits import AstroQuantity, Float, Integer
from ..binning import DefaultRecoEnergyBins
from ..cuts import EventQualitySelection
from .results import OptimizationResult

__all__ = [
    "CutOptimizerBase",
    "GhPercentileCutCalculator",
    "PercentileCuts",
    "PointSourceSensitivityOptimizer",
    "ThetaPercentileCutCalculator",
]


class CutOptimizerBase(DefaultRecoEnergyBins):
    """Base class for cut optimization algorithms."""

    needs_background = False

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

    def _check_events(self, events: dict[str, QTable]):
        if "signal" not in events.keys():
            raise ValueError(
                "Calculating G/H and/or spatial selection cuts requires 'signal' "
                f"events, but none are given. Provided events: {events.keys()}"
            )
        if self.needs_background and "background" not in events.keys():
            raise ValueError(
                "Optimizing G/H cuts for maximum point-source sensitivity "
                "requires 'background' events, but none were given. "
                f"Provided events: {events.keys()}"
            )

    @abstractmethod
    def __call__(
        self,
        events: dict[str, QTable],
        quality_query: EventQualitySelection,
        clf_prefix: str,
    ) -> OptimizationResult:
        """
        Optimize G/H (and optionally spatial selection) cuts
        and fill them in an ``OptimizationResult``.

        Parameters
        ----------
        events: dict[str, astropy.table.QTable]
            Dictionary containing tables of events used for calculating cuts.
            This has to include "signal" events and can include "background" events.
        quality_query: ctapipe.irf.EventPreprocessor
            ``ctapipe.irf.cuts.EventQualitySelection`` subclass containing preselection
            criteria for events.
        clf_prefix: str
            Prefix of the output from the G/H classifier for which the
            cut will be optimized.
        """


class GhPercentileCutCalculator(Component):
    """Computes a percentile cut on gammaness."""

    min_counts = Integer(
        default_value=10,
        help="Minimum number of events in a bin to attempt to find a cut value",
    ).tag(config=True)

    smoothing = Float(
        default_value=None,
        allow_none=True,
        help="When given, the width (in units of bins) of gaussian smoothing applied",
    ).tag(config=True)

    target_percentile = Integer(
        default_value=68,
        help="Percent of events in each energy bin to keep after the G/H cut",
    ).tag(config=True)

    def __call__(self, gammaness, reco_energy, reco_energy_bins):
        if self.smoothing and self.smoothing < 0:
            self.smoothing = None

        return calculate_percentile_cut(
            gammaness,
            reco_energy,
            reco_energy_bins,
            smoothing=self.smoothing,
            percentile=100 - self.target_percentile,
            fill_value=gammaness.max(),
            min_events=self.min_counts,
        )


class ThetaPercentileCutCalculator(Component):
    """Computes a percentile cut on theta."""

    theta_min_angle = AstroQuantity(
        default_value=u.Quantity(-1, u.deg),
        physical_type=u.physical.angle,
        help="Smallest angular cut value allowed (-1 means no cut)",
    ).tag(config=True)

    theta_max_angle = AstroQuantity(
        default_value=u.Quantity(0.32, u.deg),
        physical_type=u.physical.angle,
        help="Largest angular cut value allowed",
    ).tag(config=True)

    min_counts = Integer(
        default_value=10,
        help="Minimum number of events in a bin to attempt to find a cut value",
    ).tag(config=True)

    theta_fill_value = AstroQuantity(
        default_value=u.Quantity(0.32, u.deg),
        physical_type=u.physical.angle,
        help="Angular cut value used for bins with too few events",
    ).tag(config=True)

    smoothing = Float(
        default_value=None,
        allow_none=True,
        help="When given, the width (in units of bins) of gaussian smoothing applied",
    ).tag(config=True)

    target_percentile = Integer(
        default_value=68,
        help="Percent of events in each energy bin to keep after the theta cut",
    ).tag(config=True)

    def __call__(self, theta, reco_energy, reco_energy_bins):
        if self.theta_min_angle < 0 * u.deg:
            theta_min_angle = None
        else:
            theta_min_angle = self.theta_min_angle

        if self.theta_max_angle < 0 * u.deg:
            theta_max_angle = None
        else:
            theta_max_angle = self.theta_max_angle

        if self.smoothing and self.smoothing < 0:
            self.smoothing = None

        return calculate_percentile_cut(
            theta,
            reco_energy,
            reco_energy_bins,
            min_value=theta_min_angle,
            max_value=theta_max_angle,
            smoothing=self.smoothing,
            percentile=self.target_percentile,
            fill_value=self.theta_fill_value,
            min_events=self.min_counts,
        )


class PercentileCuts(CutOptimizerBase):
    """
    Calculates G/H separation cut based on the percentile of signal events
    to keep in each bin.
    Optionally also calculates a percentile cut on theta based on the signal
    events surviving this G/H cut.
    """

    classes = [GhPercentileCutCalculator, ThetaPercentileCutCalculator]

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.gh_cut_calculator = GhPercentileCutCalculator(parent=self)
        self.theta_cut_calculator = ThetaPercentileCutCalculator(parent=self)

    def __call__(
        self,
        events: dict[str, QTable],
        quality_query: EventQualitySelection,
        clf_prefix: str,
    ) -> OptimizationResult:
        self._check_events(events)

        gh_cuts = self.gh_cut_calculator(
            events["signal"]["gh_score"],
            events["signal"]["reco_energy"],
            self.reco_energy_bins,
        )
        gh_mask = evaluate_binned_cut(
            events["signal"]["gh_score"],
            events["signal"]["reco_energy"],
            gh_cuts,
            op=operator.ge,
        )
        spatial_selection_table = self.theta_cut_calculator(
            events["signal"]["theta"][gh_mask],
            events["signal"]["reco_energy"][gh_mask],
            self.reco_energy_bins,
        )

        result = OptimizationResult(
            quality_selection=quality_query,
            gh_cuts=gh_cuts,
            clf_prefix=clf_prefix,
            valid_energy_min=self.reco_energy_min,
            valid_energy_max=self.reco_energy_max,
            # A single set of cuts is calculated for the whole fov atm
            valid_offset_min=0 * u.deg,
            valid_offset_max=np.inf * u.deg,
            spatial_selection_table=spatial_selection_table,
        )
        return result


class PointSourceSensitivityOptimizer(CutOptimizerBase):
    """
    Optimizes a G/H cut for maximum point source sensitivity and
    calculates a percentile cut on theta.
    """

    needs_background = True
    classes = [ThetaPercentileCutCalculator]

    initial_gh_cut_efficency = Float(
        default_value=0.4, help="Start value of gamma efficiency before optimization"
    ).tag(config=True)

    max_gh_cut_efficiency = Float(
        default_value=0.8, help="Maximum gamma efficiency requested"
    ).tag(config=True)

    gh_cut_efficiency_step = Float(
        default_value=0.1,
        help="Stepsize used for scanning after optimal gammaness cut",
    ).tag(config=True)

    alpha = Float(
        default_value=0.2,
        help="Size ratio of on region / off region.",
    ).tag(config=True)

    min_background_fov_offset = AstroQuantity(
        help=(
            "Minimum distance from the fov center for background events "
            "to be taken into account"
        ),
        default_value=u.Quantity(0, u.deg),
        physical_type=u.physical.angle,
    ).tag(config=True)

    max_background_fov_offset = AstroQuantity(
        help=(
            "Maximum distance from the fov center for background events "
            "to be taken into account"
        ),
        default_value=u.Quantity(5, u.deg),
        physical_type=u.physical.angle,
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.theta_cut_calculator = ThetaPercentileCutCalculator(parent=self)

    def __call__(
        self,
        events: dict[str, QTable],
        quality_query: EventQualitySelection,
        clf_prefix: str,
    ) -> OptimizationResult:
        self._check_events(events)

        initial_gh_cuts = calculate_percentile_cut(
            events["signal"]["gh_score"],
            events["signal"]["reco_energy"],
            bins=self.reco_energy_bins,
            fill_value=0.0,
            percentile=100 * (1 - self.initial_gh_cut_efficency),
            min_events=10,
            smoothing=1,
        )
        initial_gh_mask = evaluate_binned_cut(
            events["signal"]["gh_score"],
            events["signal"]["reco_energy"],
            initial_gh_cuts,
            op=operator.gt,
        )

        spatial_selection_table = self.theta_cut_calculator(
            events["signal"]["theta"][initial_gh_mask],
            events["signal"]["reco_energy"][initial_gh_mask],
            self.reco_energy_bins,
        )
        self.log.info("Optimizing G/H separation cut for best sensitivity")

        gh_cut_efficiencies = np.arange(
            self.gh_cut_efficiency_step,
            self.max_gh_cut_efficiency + self.gh_cut_efficiency_step / 2,
            self.gh_cut_efficiency_step,
        )
        opt_sensitivity, gh_cuts = optimize_gh_cut(
            events["signal"],
            events["background"],
            reco_energy_bins=self.reco_energy_bins,
            gh_cut_efficiencies=gh_cut_efficiencies,
            op=operator.ge,
            theta_cuts=spatial_selection_table,
            alpha=self.alpha,
            fov_offset_max=self.max_background_fov_offset,
            fov_offset_min=self.min_background_fov_offset,
        )
        valid_energy = self._get_valid_energy_range(opt_sensitivity)

        # Re-calculate theta cut with optimized g/h cut
        gh_mask = evaluate_binned_cut(
            events["signal"]["gh_score"],
            events["signal"]["reco_energy"],
            gh_cuts,
            operator.ge,
        )
        events["signal"]["selected_gh"] = gh_mask
        spatial_selection_table_opt = self.theta_cut_calculator(
            events["signal"][gh_mask]["theta"],
            events["signal"][gh_mask]["reco_energy"],
            self.reco_energy_bins,
        )

        result = OptimizationResult(
            quality_selection=quality_query,
            gh_cuts=gh_cuts,
            clf_prefix=clf_prefix,
            valid_energy_min=valid_energy[0],
            valid_energy_max=valid_energy[1],
            # A single set of cuts is calculated for the whole fov atm
            valid_offset_min=self.min_background_fov_offset,
            valid_offset_max=self.max_background_fov_offset,
            spatial_selection_table=spatial_selection_table_opt,
        )
        return result

    def _get_valid_energy_range(self, opt_sensitivity):
        keep_mask = np.isfinite(opt_sensitivity["significance"])

        count = np.arange(start=0, stop=len(keep_mask), step=1)
        if all(np.diff(count[keep_mask]) == 1):
            return [
                opt_sensitivity["reco_energy_low"][keep_mask][0],
                opt_sensitivity["reco_energy_high"][keep_mask][-1],
            ]
        else:
            raise ValueError("Optimal significance curve has internal NaN bins")
