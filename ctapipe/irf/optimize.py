"""module containing optimization related functions and classes"""
import operator

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from pyirf.binning import create_bins_per_decade
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut

from ..core import Component, QualityQuery
from ..core.traits import Float


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

    def set_result(self, gh_cuts, theta_cuts, valid_energy, valid_offset):
        if not self._precuts:
            raise ValueError("Precuts must be defined before results can be saved")

        gh_cuts.meta["extname"] = "GH_CUTS"
        theta_cuts.meta["extname"] = "RAD_MAX"

        energy_lim_tab = QTable(rows=[valid_energy], names=["energy_min", "energy_max"])
        energy_lim_tab.meta["extname"] = "VALID_ENERGY"

        offset_lim_tab = QTable(rows=[valid_offset], names=["offset_min", "offset_max"])
        offset_lim_tab.meta["extname"] = "VALID_OFFSET"

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
        cut_expr_tab.meta["extname"] = "QUALITY_CUTS_EXPR"

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
        precuts = QualityQuery()
        precuts.quality_criteria = cut_expr_lst
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
        help="Number of edges per decade for Reco Energy bins",
        default_value=5,
    ).tag(config=True)

    def reco_energy_bins(self):
        """
        Creates bins per decade for reconstructed MC energy using pyirf function.
        """
        reco_energy = create_bins_per_decade(
            self.reco_energy_min * u.TeV,
            self.reco_energy_max * u.TeV,
            self.reco_energy_n_bins_per_decade,
        )
        return reco_energy

    def optimize_gh_cut(
        self,
        signal,
        background,
        alpha,
        min_fov_radius,
        max_fov_radius,
        theta,
        precuts,
    ):
        if not isinstance(max_fov_radius, u.Quantity):
            raise ValueError("max_fov_radius has to have a unit")
        if not isinstance(min_fov_radius, u.Quantity):
            raise ValueError("min_fov_radius has to have a unit")
        initial_gh_cuts = calculate_percentile_cut(
            signal["gh_score"],
            signal["reco_energy"],
            bins=self.reco_energy_bins(),
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
            self.reco_energy_bins(),
        )

        self.log.info("Optimizing G/H separation cut for best sensitivity")
        gh_cut_efficiencies = np.arange(
            self.gh_cut_efficiency_step,
            self.max_gh_cut_efficiency + self.gh_cut_efficiency_step / 2,
            self.gh_cut_efficiency_step,
        )

        opt_sens, gh_cuts = optimize_gh_cut(
            signal,
            background,
            reco_energy_bins=self.reco_energy_bins(),
            gh_cut_efficiencies=gh_cut_efficiencies,
            op=operator.ge,
            theta_cuts=theta_cuts,
            alpha=alpha,
            fov_offset_max=max_fov_radius,
            fov_offset_min=min_fov_radius,
        )
        valid_energy = self._get_valid_energy_range(opt_sens)

        result_saver = OptimizationResultStore(precuts)
        result_saver.set_result(
            gh_cuts,
            theta_cuts,
            valid_energy=valid_energy,
            valid_offset=[min_fov_radius, max_fov_radius],
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
