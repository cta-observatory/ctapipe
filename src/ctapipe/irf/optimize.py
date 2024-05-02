"""module containing optimization related functions and classes"""
import operator

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from pyirf.binning import create_bins_per_decade
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut

from ..core import Component, QualityQuery
from ..core.traits import Float, Integer


class ResultValidRange:
    def __init__(self, bounds_table, prefix):
        self.min = bounds_table[f"{prefix}_min"]
        self.max = bounds_table[f"{prefix}_max"]


class OptimizationResult:
    def __init__(self, precuts, valid_energy, valid_offset, gh, theta):
        self.precuts = precuts
        self.valid_energy = ResultValidRange(valid_energy, "energy")
        self.valid_offset = ResultValidRange(valid_offset, "offset")
        self.gh_cuts = gh
        self.theta_cuts = theta

    def __repr__(self):
        return (
            f"<OptimizationResult with {len(self.gh_cuts)} G/H bins "
            f"and {len(self.theta_cuts)} theta bins valid "
            f"between {self.valid_offset.min} to {self.valid_offset.max} "
            f"and {self.valid_energy.min} to {self.valid_energy.max} "
            f"with {len(self.precuts.quality_criteria)} precuts>"
        )


class OptimizationResultStore:
    def __init__(self, precuts=None):
        if precuts:
            if isinstance(precuts, QualityQuery):
                self._precuts = precuts.quality_criteria
                if len(self._precuts) == 0:
                    self._precuts = [(" ", " ")]  # Ensures table serialises with units
            elif isinstance(precuts, list):
                self._precuts = precuts
            else:
                self._precuts = list(precuts)
        else:
            self._precuts = None

        self._results = None

    def set_result(self, gh_cuts, theta_cuts, valid_energy, valid_offset, clf_prefix):
        if not self._precuts:
            raise ValueError("Precuts must be defined before results can be saved")

        gh_cuts.meta["EXTNAME"] = "GH_CUTS"
        gh_cuts.meta["CLFNAME"] = clf_prefix
        theta_cuts.meta["EXTNAME"] = "RAD_MAX"

        energy_lim_tab = QTable(rows=[valid_energy], names=["energy_min", "energy_max"])
        energy_lim_tab.meta["EXTNAME"] = "VALID_ENERGY"

        offset_lim_tab = QTable(rows=[valid_offset], names=["offset_min", "offset_max"])
        offset_lim_tab.meta["EXTNAME"] = "VALID_OFFSET"

        self._results = [gh_cuts, theta_cuts, energy_lim_tab, offset_lim_tab]

    def write(self, output_name, overwrite=False):
        if not isinstance(self._results, list):
            raise ValueError(
                "The results of this object"
                "have not been properly initialised,"
                " call `set_results` before writing."
            )

        cut_expr_tab = Table(
            rows=self._precuts,
            names=["name", "cut_expr"],
            dtype=[np.unicode_, np.unicode_],
        )
        cut_expr_tab.meta["EXTNAME"] = "QUALITY_CUTS_EXPR"

        cut_expr_tab.write(output_name, format="fits", overwrite=overwrite)

        for table in self._results:
            table.write(output_name, format="fits", append=True)

    def read(self, file_name):
        cut_expr_tab = Table.read(file_name, hdu=1)
        cut_expr_lst = [(name, expr) for name, expr in cut_expr_tab.iterrows()]
        # TODO: this crudely fixes a problem when loading non empty tables, make it nicer
        try:
            cut_expr_lst.remove((" ", " "))
        except ValueError:
            pass
        precuts = QualityQuery(quality_criteria=cut_expr_lst)
        gh_cuts = QTable.read(file_name, hdu=2)
        theta_cuts = QTable.read(file_name, hdu=3)
        valid_energy = QTable.read(file_name, hdu=4)
        valid_offset = QTable.read(file_name, hdu=5)

        return OptimizationResult(
            precuts, valid_energy, valid_offset, gh_cuts, theta_cuts
        )


class GridOptimizer(Component):
    """Performs cut optimization"""

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

    reco_energy_min = Float(
        help="Minimum value for Reco Energy bins in TeV units",
        default_value=0.015,
    ).tag(config=True)

    reco_energy_max = Float(
        help="Maximum value for Reco Energy bins in TeV units",
        default_value=200,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Float(
        help="Number of bins per decade for Reco Energy bins",
        default_value=5,
    ).tag(config=True)

    def optimize_gh_cut(
        self,
        signal,
        background,
        alpha,
        min_fov_radius,
        max_fov_radius,
        theta,
        precuts,
        clf_prefix,
        point_like,
    ):
        if not isinstance(max_fov_radius, u.Quantity):
            raise ValueError("max_fov_radius has to have a unit")
        if not isinstance(min_fov_radius, u.Quantity):
            raise ValueError("min_fov_radius has to have a unit")

        reco_energy_bins = create_bins_per_decade(
            self.reco_energy_min * u.TeV,
            self.reco_energy_max * u.TeV,
            self.reco_energy_n_bins_per_decade,
        )

        if point_like:
            initial_gh_cuts = calculate_percentile_cut(
                signal["gh_score"],
                signal["reco_energy"],
                bins=reco_energy_bins,
                fill_value=0.0,
                percentile=100 * (1 - self.initial_gh_cut_efficency),
                min_events=25,
                smoothing=1,
            )
            initial_gh_mask = evaluate_binned_cut(
                signal["gh_score"],
                signal["reco_energy"],
                initial_gh_cuts,
                op=operator.gt,
            )

            theta_cuts = theta.calculate_theta_cuts(
                signal["theta"][initial_gh_mask],
                signal["reco_energy"][initial_gh_mask],
                reco_energy_bins,
            )
        else:
            # TODO: Find a better solution for full enclosure than this dummy theta cut
            self.log.info("Optimizing G/H separation cut without prior theta cut.")
            theta_cuts = QTable()
            theta_cuts["low"] = reco_energy_bins[:-1]
            theta_cuts["center"] = 0.5 * (reco_energy_bins[:-1] + reco_energy_bins[1:])
            theta_cuts["high"] = reco_energy_bins[1:]
            theta_cuts["cut"] = max_fov_radius

        self.log.info("Optimizing G/H separation cut for best sensitivity")
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
            alpha=alpha,
            fov_offset_max=max_fov_radius,
            fov_offset_min=min_fov_radius,
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
            theta_cuts_opt = theta.calculate_theta_cuts(
                signal[signal["selected_gh"]]["theta"],
                signal[signal["selected_gh"]]["reco_energy"],
            )
        else:
            # TODO: Find a better solution for full enclosure than this dummy theta cut
            theta_cuts_opt = QTable()
            theta_cuts_opt["low"] = reco_energy_bins[:-1]
            theta_cuts_opt["center"] = 0.5 * (
                reco_energy_bins[:-1] + reco_energy_bins[1:]
            )
            theta_cuts_opt["high"] = reco_energy_bins[1:]
            theta_cuts_opt["cut"] = max_fov_radius

        result_saver = OptimizationResultStore(precuts)
        result_saver.set_result(
            gh_cuts,
            theta_cuts_opt,
            valid_energy=valid_energy,
            valid_offset=[min_fov_radius, max_fov_radius],
            clf_prefix=clf_prefix,
        )

        return result_saver, opt_sens

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


class ThetaCutsCalculator(Component):
    """Compute percentile cuts on theta"""

    theta_min_angle = Float(
        default_value=-1, help="Smallest angular cut value allowed (-1 means no cut)"
    ).tag(config=True)

    theta_max_angle = Float(
        default_value=0.32, help="Largest angular cut value allowed"
    ).tag(config=True)

    theta_min_counts = Integer(
        default_value=10,
        help="Minimum number of events in a bin to attempt to find a cut value",
    ).tag(config=True)

    theta_fill_value = Float(
        default_value=0.32, help="Angular cut value used for bins with too few events"
    ).tag(config=True)

    theta_smoothing = Float(
        default_value=None,
        allow_none=True,
        help="When given, the width (in units of bins) of gaussian smoothing applied (None)",
    ).tag(config=True)

    target_percentile = Float(
        default_value=68,
        help="Percent of events in each energy bin to keep after the theta cut",
    ).tag(config=True)

    reco_energy_min = Float(
        help="Minimum value for Reco Energy bins in TeV units",
        default_value=0.015,
    ).tag(config=True)

    reco_energy_max = Float(
        help="Maximum value for Reco Energy bins in TeV units",
        default_value=200,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Float(
        help="Number of bins per decade for Reco Energy bins",
        default_value=5,
    ).tag(config=True)

    def calculate_theta_cuts(self, theta, reco_energy, reco_energy_bins=None):
        if reco_energy_bins is None:
            reco_energy_bins = create_bins_per_decade(
                self.reco_energy_min * u.TeV,
                self.reco_energy_max * u.TeV,
                self.reco_energy_n_bins_per_decade,
            )

        theta_min_angle = (
            None if self.theta_min_angle < 0 else self.theta_min_angle * u.deg
        )
        theta_max_angle = (
            None if self.theta_max_angle < 0 else self.theta_max_angle * u.deg
        )
        if self.theta_smoothing:
            theta_smoothing = None if self.theta_smoothing < 0 else self.theta_smoothing
        else:
            theta_smoothing = self.theta_smoothing

        return calculate_percentile_cut(
            theta,
            reco_energy,
            reco_energy_bins,
            min_value=theta_min_angle,
            max_value=theta_max_angle,
            smoothing=theta_smoothing,
            percentile=self.target_percentile,
            fill_value=self.theta_fill_value * u.deg,
            min_events=self.theta_min_counts,
        )
