"""module containing optimization related functions and classes"""

import operator
from abc import abstractmethod
from collections.abc import Sequence

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import QTable, Table
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut

from ..core import Component, QualityQuery
from ..core.traits import AstroQuantity, Float, Integer, Path
from .binning import ResultValidRange, make_bins_per_decade
from .select import EventPreProcessor


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
        theta_cuts: QTable | None = None,
        precuts: QualityQuery | Sequence | None = None,
    ) -> None:
        if precuts:
            if isinstance(precuts, QualityQuery):
                if len(precuts.quality_criteria) == 0:
                    precuts.quality_criteria = [
                        (" ", " ")
                    ]  # Ensures table serialises properly

                self.precuts = precuts
            elif isinstance(precuts, list):
                self.precuts = QualityQuery(quality_criteria=precuts)
            else:
                self.precuts = QualityQuery(quality_criteria=list(precuts))
        else:
            self.precuts = QualityQuery(quality_criteria=[(" ", " ")])

        self.valid_energy = ResultValidRange(min=valid_energy_min, max=valid_energy_max)
        self.valid_offset = ResultValidRange(min=valid_offset_min, max=valid_offset_max)
        self.gh_cuts = gh_cuts
        self.clf_prefix = clf_prefix
        self.theta_cuts = theta_cuts

    def __repr__(self):
        if self.theta_cuts is not None:
            return (
                f"<OptimizationResult with {len(self.gh_cuts)} G/H bins "
                f"and {len(self.theta_cuts)} theta bins valid "
                f"between {self.valid_offset.min} to {self.valid_offset.max} "
                f"and {self.valid_energy.min} to {self.valid_energy.max} "
                f"with {len(self.precuts.quality_criteria)} precuts>"
            )
        else:
            return (
                f"<OptimizationResult with {len(self.gh_cuts)} G/H bins valid "
                f"between {self.valid_offset.min} to {self.valid_offset.max} "
                f"and {self.valid_energy.min} to {self.valid_energy.max} "
                f"with {len(self.precuts.quality_criteria)} precuts>"
            )

    def write(self, output_name: Path | str, overwrite: bool = False) -> None:
        """Write an ``OptimizationResult`` to a file in FITS format."""

        cut_expr_tab = Table(
            rows=self.precuts.quality_criteria,
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

        if self.theta_cuts is not None:
            self.theta_cuts.meta["EXTNAME"] = "RAD_MAX"
            results.append(self.theta_cuts)

        # Overwrite if needed and allowed
        results[0].write(output_name, format="fits", overwrite=overwrite)

        for table in results[1:]:
            table.write(output_name, format="fits", append=True)

    @classmethod
    def read(cls, file_name):
        """Read an ``OptimizationResult`` from a file in FITS format."""

        with fits.open(file_name) as hdul:
            cut_expr_tab = Table.read(hdul[1])
            cut_expr_lst = [(name, expr) for name, expr in cut_expr_tab.iterrows()]
            if (" ", " ") in cut_expr_lst:
                cut_expr_lst.remove((" ", " "))

            precuts = QualityQuery(quality_criteria=cut_expr_lst)
            gh_cuts = QTable.read(hdul[2])
            valid_energy = QTable.read(hdul[3])
            valid_offset = QTable.read(hdul[4])
            theta_cuts = QTable.read(hdul[5]) if len(hdul) > 5 else None

        return cls(
            precuts=precuts,
            valid_energy_min=valid_energy["energy_min"],
            valid_energy_max=valid_energy["energy_max"],
            valid_offset_min=valid_offset["offset_min"],
            valid_offset_max=valid_offset["offset_max"],
            gh_cuts=gh_cuts,
            clf_prefix=gh_cuts.meta["CLFNAME"],
            theta_cuts=theta_cuts,
        )


class CutOptimizerBase(Component):
    """Base class for cut optimization algorithms."""

    reco_energy_min = AstroQuantity(
        help="Minimum value for Reco Energy bins",
        default_value=u.Quantity(0.015, u.TeV),
        physical_type=u.physical.energy,
    ).tag(config=True)

    reco_energy_max = AstroQuantity(
        help="Maximum value for Reco Energy bins",
        default_value=u.Quantity(150, u.TeV),
        physical_type=u.physical.energy,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Integer(
        help="Number of bins per decade for Reco Energy bins",
        default_value=5,
    ).tag(config=True)

    min_bkg_fov_offset = AstroQuantity(
        help=(
            "Minimum distance from the fov center for background events "
            "to be taken into account"
        ),
        default_value=u.Quantity(0, u.deg),
        physical_type=u.physical.angle,
    ).tag(config=True)

    max_bkg_fov_offset = AstroQuantity(
        help=(
            "Maximum distance from the fov center for background events "
            "to be taken into account"
        ),
        default_value=u.Quantity(5, u.deg),
        physical_type=u.physical.angle,
    ).tag(config=True)

    @abstractmethod
    def optimize_cuts(
        self,
        signal: QTable,
        background: QTable,
        precuts: EventPreProcessor,
        clf_prefix: str,
        point_like: bool,
    ) -> OptimizationResult:
        """
        Optimize G/H (and optionally theta) cuts
        and fill them in an ``OptimizationResult``.

        Parameters
        ----------
        signal: astropy.table.QTable
            Table containing signal events
        background: astropy.table.QTable
            Table containing background events
        precuts: ctapipe.irf.EventPreProcessor
            ``ctapipe.core.QualityQuery`` subclass containing preselection
            criteria for events
        clf_prefix: str
            Prefix of the output from the G/H classifier for which the
            cut will be optimized
        point_like: bool
            Whether a theta cut should be calculated (True) or only a
            G/H cut (False)
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

    def calculate_gh_cut(self, gammaness, reco_energy, reco_energy_bins):
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

    def calculate_theta_cut(self, theta, reco_energy, reco_energy_bins):
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

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.gh = GhPercentileCutCalculator(parent=self)
        self.theta = ThetaPercentileCutCalculator(parent=self)

    def optimize_cuts(
        self,
        signal: QTable,
        background: QTable,
        precuts: EventPreProcessor,
        clf_prefix: str,
        point_like: bool,
    ) -> OptimizationResult:
        reco_energy_bins = make_bins_per_decade(
            self.reco_energy_min.to(u.TeV),
            self.reco_energy_max.to(u.TeV),
            self.reco_energy_n_bins_per_decade,
        )
        gh_cuts = self.gh.calculate_gh_cut(
            signal["gh_score"],
            signal["reco_energy"],
            reco_energy_bins,
        )
        if point_like:
            gh_mask = evaluate_binned_cut(
                signal["gh_score"],
                signal["reco_energy"],
                gh_cuts,
                op=operator.ge,
            )
            theta_cuts = self.theta.calculate_theta_cut(
                signal["theta"][gh_mask],
                signal["reco_energy"][gh_mask],
                reco_energy_bins,
            )

        result = OptimizationResult(
            precuts=precuts,
            gh_cuts=gh_cuts,
            clf_prefix=clf_prefix,
            valid_energy_min=self.reco_energy_min,
            valid_energy_max=self.reco_energy_max,
            # A single set of cuts is calculated for the whole fov atm
            valid_offset_min=0 * u.deg,
            valid_offset_max=np.inf * u.deg,
            theta_cuts=theta_cuts if point_like else None,
        )
        return result


class PointSourceSensitivityOptimizer(CutOptimizerBase):
    """
    Optimizes a G/H cut for maximum point source sensitivity and
    calculates a percentile cut on theta.
    """

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

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.theta = ThetaPercentileCutCalculator(parent=self)

    def optimize_cuts(
        self,
        signal: QTable,
        background: QTable,
        precuts: EventPreProcessor,
        clf_prefix: str,
        point_like: bool,
    ) -> OptimizationResult:
        reco_energy_bins = make_bins_per_decade(
            self.reco_energy_min.to(u.TeV),
            self.reco_energy_max.to(u.TeV),
            self.reco_energy_n_bins_per_decade,
        )

        if point_like:
            initial_gh_cuts = calculate_percentile_cut(
                signal["gh_score"],
                signal["reco_energy"],
                bins=reco_energy_bins,
                fill_value=0.0,
                percentile=100 * (1 - self.initial_gh_cut_efficency),
                min_events=10,
                smoothing=1,
            )
            initial_gh_mask = evaluate_binned_cut(
                signal["gh_score"],
                signal["reco_energy"],
                initial_gh_cuts,
                op=operator.gt,
            )

            theta_cuts = self.theta.calculate_theta_cut(
                signal["theta"][initial_gh_mask],
                signal["reco_energy"][initial_gh_mask],
                reco_energy_bins,
            )
            self.log.info("Optimizing G/H separation cut for best sensitivity")
        else:
            # Create a dummy theta cut since `pyirf.cut_optimization.optimize_gh_cut`
            # needs a theta cut atm.
            theta_cuts = QTable()
            theta_cuts["low"] = reco_energy_bins[:-1]
            theta_cuts["center"] = 0.5 * (reco_energy_bins[:-1] + reco_energy_bins[1:])
            theta_cuts["high"] = reco_energy_bins[1:]
            theta_cuts["cut"] = self.max_bkg_fov_offset
            self.log.info(
                "Optimizing G/H separation cut for best sensitivity "
                "with `max_bkg_fov_offset` as theta cut."
            )

        gh_cut_efficiencies = np.arange(
            self.gh_cut_efficiency_step,
            self.max_gh_cut_efficiency + self.gh_cut_efficiency_step / 2,
            self.gh_cut_efficiency_step,
        )
        opt_sens, gh_cuts = optimize_gh_cut(
            signal,
            background,
            reco_energy_bins=reco_energy_bins,
            gh_cut_efficiencies=gh_cut_efficiencies,
            op=operator.ge,
            theta_cuts=theta_cuts,
            alpha=self.alpha,
            fov_offset_max=self.max_bkg_fov_offset,
            fov_offset_min=self.min_bkg_fov_offset,
        )
        valid_energy = self._get_valid_energy_range(opt_sens)

        # Re-calculate theta cut with optimized g/h cut
        if point_like:
            signal["selected_gh"] = evaluate_binned_cut(
                signal["gh_score"],
                signal["reco_energy"],
                gh_cuts,
                operator.ge,
            )
            theta_cuts_opt = self.theta.calculate_theta_cut(
                signal[signal["selected_gh"]]["theta"],
                signal[signal["selected_gh"]]["reco_energy"],
                reco_energy_bins,
            )

        result = OptimizationResult(
            precuts=precuts,
            gh_cuts=gh_cuts,
            clf_prefix=clf_prefix,
            valid_energy_min=valid_energy[0],
            valid_energy_max=valid_energy[1],
            # A single set of cuts is calculated for the whole fov atm
            valid_offset_min=self.min_bkg_fov_offset,
            valid_offset_max=self.max_bkg_fov_offset,
            theta_cuts=theta_cuts_opt if point_like else None,
        )
        return result

    def _get_valid_energy_range(self, opt_sens):
        keep_mask = np.isfinite(opt_sens["significance"])

        count = np.arange(start=0, stop=len(keep_mask), step=1)
        if all(np.diff(count[keep_mask]) == 1):
            return [
                opt_sens["reco_energy_low"][keep_mask][0],
                opt_sens["reco_energy_high"][keep_mask][-1],
            ]
        else:
            raise ValueError("Optimal significance curve has internal NaN bins")
