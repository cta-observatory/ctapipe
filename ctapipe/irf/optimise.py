import operator

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from pyirf.binning import create_bins_per_decade
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut

from ..core import Component, QualityQuery
from ..core.traits import Float


class OptimisationResult:
    def __init__(self, gh_cuts=None, offset_lim=None):
        self.gh_cuts = gh_cuts
        if gh_cuts:
            self.gh_cuts.meta["extname"] = "GH_CUTS"
        if offset_lim and isinstance(offset_lim[0], list):
            self.offset_lim = offset_lim
        else:
            self.offset_lim = [offset_lim]

    def write(self, out_name, precuts, overwrite=False):
        if isinstance(precuts, QualityQuery):
            precuts = precuts.quality_criteria
            if len(precuts) == 0:
                precuts = [(" ", " ")]  # Ensures table can be created
        cut_expr_tab = Table(
            rows=precuts,
            names=["name", "cut_expr"],
            dtype=[np.unicode_, np.unicode_],
        )
        cut_expr_tab.meta["extname"] = "QUALITY_CUTS_EXPR"
        offset_lim_tab = QTable(
            rows=self.offset_lim, names=["offset_min", "offset_max"]
        )
        offset_lim_tab.meta["extname"] = "OFFSET_LIMITS"
        self.gh_cuts.write(out_name, format="fits", overwrite=overwrite)
        cut_expr_tab.write(out_name, format="fits", append=True)
        offset_lim_tab.write(out_name, format="fits", append=True)

    def read(self, file_name):
        self.gh_cuts = QTable.read(file_name, hdu=1)
        cut_expr_tab = Table.read(file_name, hdu=2)
        cut_expr_lst = [(name, expr) for name, expr in cut_expr_tab.iterrows()]
        # TODO: this crudely fixes a problem when loading non empty tables, make it nicer
        try:
            cut_expr_lst.remove((" ", " "))
        except ValueError:
            pass
        precuts = QualityQuery()
        precuts.quality_criteria = cut_expr_lst
        offset_lim_tab = QTable.read(file_name, hdu=3)
        # TODO: find some way to do this cleanly
        offset_lim_tab["bins"] = np.array(
            [offset_lim_tab["offset_min"], offset_lim_tab["offset_max"]]
        ).T
        self.offset_lim = (
            np.array(offset_lim_tab[0]) * offset_lim_tab["offset_max"].unit
        )
        return precuts

    def __repr__(self):
        return f"<OptimisationResult in {len(self.gh_cuts)} bins for {self.offset_lim[0]} to {self.offset_lim[1]} with {len(self.precuts.quality_criteria)} precuts>"


class GridOptimizer(Component):
    """Performs cut optimisation"""

    initial_gh_cut_efficency = Float(
        default_value=0.4, help="Start value of gamma efficiency before optimisation"
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
        default_value=0.005,
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

    def optimise_gh_cut(
        self, signal, background, alpha, min_fov_radius, max_fov_radius, theta
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

        sens2, gh_cuts = optimize_gh_cut(
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

        result = OptimisationResult(
            gh_cuts, offset_lim=[min_fov_radius, max_fov_radius]
        )

        return result, sens2
