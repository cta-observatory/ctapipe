"""module containing optimization related functions and classes"""

import operator
from abc import abstractmethod
from collections.abc import Sequence

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import QTable, Table
from pyirf.cut_optimization import optimize_cuts, optimize_gh_cut
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut

from ..core import Component, QualityQuery
from ..core.traits import AstroQuantity, Float, Integer, List, Path
from ..io.dl2_tables_preprocessing import DL2EventQualityQuery
from .binning import DefaultRecoEnergyBins, ResultValidRange

__all__ = [
    "CutOptimizerBase",
    "GhPercentileCutCalculator",
    "OptimizationResult",
    "PercentileCuts",
    "PointSourceSensitivityGhOptimizer",
    "PointSourceSensitivityOptimizer",
    "PointSourceSensitivityOptimizerBase",
    "ThetaPercentileCutCalculator",
]


class OptimizationResult:
    """Result of an optimization of G/H and theta cuts or only G/H cuts."""

    def __init__(
        self,
        valid_energy_min: u.Quantity,
        valid_energy_max: u.Quantity,
        valid_offset_min: u.Quantity,
        valid_offset_max: u.Quantity,
        gh_cuts: QTable,
        clf_prefix: str,
        quality_query: QualityQuery | Sequence | None = None,
        spatial_selection_table: QTable | None = None,
        multiplicity_cuts: QTable | None = None,
    ) -> None:
        if quality_query:
            if isinstance(quality_query, QualityQuery):
                if len(quality_query.quality_criteria) == 0:
                    quality_query.quality_criteria = [
                        (" ", " ")
                    ]  # Ensures table serialises properly

                self.quality_query = quality_query
            elif isinstance(quality_query, list):
                self.quality_query = QualityQuery(quality_criteria=quality_query)
            else:
                self.quality_query = QualityQuery(quality_criteria=list(quality_query))
        else:
            self.quality_query = QualityQuery(quality_criteria=[(" ", " ")])

        self.valid_energy = ResultValidRange(min=valid_energy_min, max=valid_energy_max)
        self.valid_offset = ResultValidRange(min=valid_offset_min, max=valid_offset_max)
        self.gh_cuts = gh_cuts
        self.clf_prefix = clf_prefix
        self.spatial_selection_table = spatial_selection_table
        self.multiplicity_cuts = multiplicity_cuts

    def __repr__(self):
        repr = f"<OptimizationResult with {len(self.gh_cuts)} G/H bins"

        if self.spatial_selection_table is not None:
            repr += f", and {len(self.spatial_selection_table)} theta bins"

        if self.multiplicity_cuts is not None:
            repr += f", and {len(self.multiplicity_cuts)} multiplicity bins"

        repr += (
            f" valid between {self.valid_offset.min} to {self.valid_offset.max}"
            f" and {self.valid_energy.min} to {self.valid_energy.max}"
            f" with {len(self.quality_query.quality_criteria)} quality criteria>"
        )
        return repr

    def write(self, output_name: Path | str, overwrite: bool = False) -> None:
        """Write an ``OptimizationResult`` to a file in FITS format."""

        cut_expr_tab = Table(
            rows=self.quality_query.quality_criteria,
            names=["name", "cut_expr"],
            dtype=[np.str_, np.str_],
        )
        cut_expr_tab.meta["EXTNAME"] = "QUALITY_CUTS_EXPR"

        self.gh_cuts.meta["EXTNAME"] = "GH_CUTS"
        self.gh_cuts.meta["CLFNAME"] = self.clf_prefix

        energy_lim_tab = QTable(
            rows=[[self.valid_energy.min, self.valid_energy.max]],
            names=["energy_min", "energy_max"],
        )
        energy_lim_tab.meta["EXTNAME"] = "VALID_ENERGY"

        offset_lim_tab = QTable(
            rows=[[self.valid_offset.min, self.valid_offset.max]],
            names=["offset_min", "offset_max"],
        )
        offset_lim_tab.meta["EXTNAME"] = "VALID_OFFSET"

        results = [cut_expr_tab, self.gh_cuts, energy_lim_tab, offset_lim_tab]

        if self.spatial_selection_table is not None:
            self.spatial_selection_table.meta["EXTNAME"] = "RAD_MAX"
            results.append(self.spatial_selection_table)

        if self.multiplicity_cuts is not None:
            self.multiplicity_cuts.meta["EXTNAME"] = "MULTIPLICITY_CUTS"
            results.append(self.multiplicity_cuts)

        primary = fits.PrimaryHDU()
        hdus = [primary]
        for table in results:
            hdus.append(fits.BinTableHDU(table))
        fits.HDUList(hdus).writeto(output_name, overwrite=overwrite)

    @classmethod
    def read(cls, file_name):
        """Read an ``OptimizationResult`` from a file in FITS format."""

        with fits.open(file_name) as hdul:
            cut_expr_tab = Table.read(hdul[1])
            cut_expr_lst = [(name, expr) for name, expr in cut_expr_tab.iterrows()]
            if (" ", " ") in cut_expr_lst:
                cut_expr_lst.remove((" ", " "))

            quality_query = QualityQuery(quality_criteria=cut_expr_lst)
            gh_cuts = QTable.read(hdul[2])
            valid_energy = QTable.read(hdul[3])
            valid_offset = QTable.read(hdul[4])
            spatial_selection_table = None
            multiplicity_cuts = None
            if len(hdul) == 6:
                if hdul[5].header["EXTNAME"] == "RAD_MAX":
                    spatial_selection_table = QTable.read(hdul[5])
                else:
                    multiplicity_cuts = QTable.read(hdul[5])
            elif len(hdul) == 7:
                spatial_selection_table = QTable.read(hdul[5])
                multiplicity_cuts = QTable.read(hdul[6])

        return cls(
            quality_query=quality_query,
            valid_energy_min=valid_energy["energy_min"],
            valid_energy_max=valid_energy["energy_max"],
            valid_offset_min=valid_offset["offset_min"],
            valid_offset_max=valid_offset["offset_max"],
            gh_cuts=gh_cuts,
            clf_prefix=gh_cuts.meta["CLFNAME"],
            spatial_selection_table=spatial_selection_table,
            multiplicity_cuts=multiplicity_cuts,
        )


class CutOptimizerBase(DefaultRecoEnergyBins):
    """Base class for cut optimization algorithms."""

    needs_background = False

    def _check_events(self, events: dict[str, QTable]):
        """To be called at the start `__call__` in all subclasses."""
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
        quality_query: DL2EventQualityQuery,
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
        quality_query: ctapipe.io.DL2EventPreprocessor
            ``ctapipe.core.QualityQuery`` subclass containing preselection
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

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.gh_cut_calculator = GhPercentileCutCalculator(parent=self)
        self.theta_cut_calculator = ThetaPercentileCutCalculator(parent=self)

    def __call__(
        self,
        events: dict[str, QTable],
        quality_query: DL2EventQualityQuery,
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
            quality_query=quality_query,
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


class PointSourceSensitivityOptimizerBase(CutOptimizerBase):
    """
    Base class for optimizers optimizing cuts for maximum point source efficiency.
    This holds some configurable traits shared between different optimization
    algorithms implemented in subclasses of this class.
    """

    needs_background = True

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


class PointSourceSensitivityGhOptimizer(PointSourceSensitivityOptimizerBase):
    """
    EventDisplay-like procedure of optimizing a G/H cut for maximum point source
    sensitivity, calculating only a percentile cut on theta, and no cut on
    multiplicity.
    """

    initial_gh_cut_efficency = Float(
        default_value=0.4, help="Start value of gamma efficiency before optimization"
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.theta_cut_calculator = ThetaPercentileCutCalculator(parent=self)

    def __call__(
        self,
        events: dict[str, QTable],
        quality_query: DL2EventQualityQuery,
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
            quality_query=quality_query,
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


class PointSourceSensitivityOptimizer(PointSourceSensitivityOptimizerBase):
    """
    Finds the combination of G/H cut, theta cut, and cut on the event-multiplicity
    for each energy bin with the maximum point source sensitivity in a grid search.
    """

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

    max_theta_cut_efficiency = Float(
        default_value=0.8, help="Maximum gamma efficiency of the theta cut requested"
    ).tag(config=True)

    theta_cut_efficiency_step = Float(
        default_value=0.1,
        help="Efficiency-stepsize used for scanning after optimal theta cut",
    ).tag(config=True)

    multiplicity_cuts = List(
        Integer(),
        default_value=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Event-multiplicity cuts used for scanning after optimal cut",
    ).tag(config=True)

    def __call__(
        self,
        events: dict[str, QTable],
        quality_query: EventQualityQuery,
        clf_prefix: str,
    ) -> OptimizationResult:
        self._check_events(events)

        gh_cut_efficiencies = np.arange(
            self.gh_cut_efficiency_step,
            self.max_gh_cut_efficiency + self.gh_cut_efficiency_step / 2,
            self.gh_cut_efficiency_step,
        )
        theta_cut_efficiencies = np.arange(
            self.theta_cut_efficiency_step,
            self.max_theta_cut_efficiency + self.theta_cut_efficiency_step / 2,
            self.theta_cut_efficiency_step,
        )

        opt_sensitivity, multiplicity_cuts, theta_cuts, gh_cuts = optimize_cuts(
            events["signal"],
            events["background"],
            reco_energy_bins=self.reco_energy_bins,
            multiplicity_cuts=self.multiplicity_cuts,
            gh_cut_efficiencies=gh_cut_efficiencies,
            theta_cut_efficiencies=theta_cut_efficiencies,
            theta_min_value=self.theta_min_angle,
            theta_max_value=self.theta_max_angle,
            alpha=self.alpha,
            fov_offset_max=self.max_background_fov_offset,
            fov_offset_min=self.min_background_fov_offset,
        )
        valid_energy = self._get_valid_energy_range(opt_sensitivity)

        result = OptimizationResult(
            quality_query=quality_query,
            clf_prefix=clf_prefix,
            gh_cuts=gh_cuts,
            valid_energy_min=valid_energy[0],
            valid_energy_max=valid_energy[1],
            # A single set of cuts is calculated for the whole fov atm
            valid_offset_min=self.min_background_fov_offset,
            valid_offset_max=self.max_background_fov_offset,
            spatial_selection_table=theta_cuts,
            multiplicity_cuts=multiplicity_cuts,
        )
        return result
