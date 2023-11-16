import operator

import astropy.units as u
import numpy as np
from pyirf.binning import create_bins_per_decade
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut

from ..core import Component
from ..core.traits import Float


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
            fov_offset_max=max_fov_radius * u.deg,
            fov_offset_min=min_fov_radius * u.deg,
        )

        # now that we have the optimized gh cuts, we recalculate the theta
        # cut as 68 percent containment on the events surviving these cuts.
        for tab in (signal, background):
            tab["selected_gh"] = evaluate_binned_cut(
                tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
            )
        self.log.info("Recalculating theta cut for optimized GH Cuts")

        theta_cuts = theta.calculate_theta_cuts(
            signal[signal["selected_gh"]]["theta"],
            signal[signal["selected_gh"]]["reco_energy"],
            self.reco_energy_bins(),
        )

        return gh_cuts, theta_cuts, sens2
