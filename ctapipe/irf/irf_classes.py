"""
Define a parent IrfTool class to hold all the options
"""
import operator

import astropy.units as u
import numpy as np
from astropy.table import QTable
from pyirf.binning import create_bins_per_decade
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut

from ..core import Component, QualityQuery
from ..core.traits import Float, Integer, List, Unicode


class CutOptimizer(Component):
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


class ThetaCutsCalculator(Component):
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
        default_value=-1,
        help="When given, the width (in units of bins) of gaussian smoothing applied (-1)",
    ).tag(config=True)

    target_percentile = Float(
        default_value=68,
        help="Percent of events in each energy bin keep after the theta cut",
    ).tag(config=True)

    def calculate_theta_cuts(self, theta, reco_energy, energy_bins):
        theta_min_angle = (
            None if self.theta_min_angle < 0 else self.theta_min_angle * u.deg
        )
        theta_max_angle = (
            None if self.theta_max_angle < 0 else self.theta_max_angle * u.deg
        )
        theta_smoothing = None if self.theta_smoothing < 0 else self.theta_smoothing

        return calculate_percentile_cut(
            theta,
            reco_energy,
            energy_bins,
            min_value=theta_min_angle,
            max_value=theta_max_angle,
            smoothing=theta_smoothing,
            percentile=self.target_percentile,
            fill_value=self.theta_fill_value * u.deg,
            min_events=self.theta_min_counts,
        )


class EventPreProcessor(QualityQuery):
    """Defines preselection cuts and the necessary renaming of columns"""

    energy_reconstructor = Unicode(
        default_value="RandomForestRegressor",
        help="Prefix of the reco `_energy` column",
    ).tag(config=True)
    geometry_reconstructor = Unicode(
        default_value="HillasReconstructor",
        help="Prefix of the `_alt` and `_az` reco geometry columns",
    ).tag(config=True)
    gammaness_classifier = Unicode(
        default_value="RandomForestClassifier",
        help="Prefix of the classifier `_prediction` column",
    ).tag(config=True)

    quality_criteria = List(
        default_value=[
            ("multiplicity 4", "np.count_nonzero(tels_with_trigger,axis=1) >= 4"),
            ("valid classifier", "RandomForestClassifier_is_valid"),
            ("valid geom reco", "HillasReconstructor_is_valid"),
            ("valid energy reco", "RandomForestRegressor_is_valid"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)

    rename_columns = List(
        help="List containing translation pairs new and old column names"
        "used when processing input with names differing from the CTA prod5b format"
        "Ex: [('valid_geom','HillasReconstructor_is_valid')]",
        default_value=[],
    ).tag(config=True)

    def normalise_column_names(self, events):
        keep_columns = [
            "obs_id",
            "event_id",
            "true_energy",
            "true_az",
            "true_alt",
        ]
        rename_from = [
            f"{self.energy_reconstructor}_energy",
            f"{self.geometry_reconstructor}_az",
            f"{self.geometry_reconstructor}_alt",
            f"{self.gammaness_classifier}_prediction",
        ]
        rename_to = ["reco_energy", "reco_az", "reco_alt", "gh_score"]

        # We never enter the loop if rename_columns is empty
        for new, old in self.rename_columns:
            rename_from.append(old)
            rename_to.append(new)

        keep_columns.extend(rename_from)
        events = QTable(events[keep_columns], copy=False)
        events.rename_columns(rename_from, rename_to)
        return events

    def make_empty_table(self):
        """This function defines the columns later functions expect to be present in the event table"""
        columns = [
            "obs_id",
            "event_id",
            "true_energy",
            "true_az",
            "true_alt",
            "reco_energy",
            "reco_az",
            "reco_alt",
            "gh_score",
            "pointing_az",
            "pointing_alt",
            "theta",
            "true_source_fov_offset",
            "reco_source_fov_offset",
            "weight",
        ]
        units = {
            "true_energy": u.TeV,
            "true_az": u.deg,
            "true_alt": u.deg,
            "reco_energy": u.TeV,
            "reco_az": u.deg,
            "reco_alt": u.deg,
            "pointing_az": u.deg,
            "pointing_alt": u.deg,
            "theta": u.deg,
            "true_source_fov_offset": u.deg,
            "reco_source_fov_offset": u.deg,
        }

        return QTable(names=columns, units=units)


class OutputEnergyBinning(Component):
    """Collects energy binning settings"""

    true_energy_min = Float(
        help="Minimum value for True Energy bins in TeV units",
        default_value=0.005,
    ).tag(config=True)

    true_energy_max = Float(
        help="Maximum value for True Energy bins in TeV units",
        default_value=200,
    ).tag(config=True)

    true_energy_n_bins_per_decade = Float(
        help="Number of edges per decade for True Energy bins",
        default_value=10,
    ).tag(config=True)

    reco_energy_min = Float(
        help="Minimum value for Reco Energy bins in TeV units",
        default_value=0.006,
    ).tag(config=True)

    reco_energy_max = Float(
        help="Maximum value for Reco Energy bins in TeV units",
        default_value=190,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Float(
        help="Number of edges per decade for Reco Energy bins",
        default_value=5,
    ).tag(config=True)

    energy_migration_min = Float(
        help="Minimum value of Energy Migration matrix",
        default_value=0.2,
    ).tag(config=True)

    energy_migration_max = Float(
        help="Maximum value of Energy Migration matrix",
        default_value=5,
    ).tag(config=True)

    energy_migration_n_bins = Integer(
        help="Number of bins in log scale for Energy Migration matrix",
        default_value=31,
    ).tag(config=True)

    def true_energy_bins(self):
        """
        Creates bins per decade for true MC energy using pyirf function.
        """
        true_energy = create_bins_per_decade(
            self.true_energy_min * u.TeV,
            self.true_energy_max * u.TeV,
            self.true_energy_n_bins_per_decade,
        )
        return true_energy

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

    def energy_migration_bins(self):
        """
        Creates bins for energy migration.
        """
        energy_migration = np.geomspace(
            self.energy_migration_min,
            self.energy_migration_max,
            self.energy_migration_n_bins,
        )
        return energy_migration


class DataBinning(Component):
    """
    Collects information on generating energy and angular bins for
    generating IRFs as per pyIRF requirements.

    Stolen from LSTChain
    """

    fov_offset_min = Float(
        help="Minimum value for FoV Offset bins in degrees",
        default_value=0.0,
    ).tag(config=True)

    fov_offset_max = Float(
        help="Maximum value for FoV offset bins in degrees",
        default_value=5.0,
    ).tag(config=True)

    fov_offset_n_edges = Integer(
        help="Number of edges for FoV offset bins",
        default_value=2,
    ).tag(config=True)

    source_offset_min = Float(
        help="Minimum value for Source offset for PSF IRF",
        default_value=0,
    ).tag(config=True)

    source_offset_max = Float(
        help="Maximum value for Source offset for PSF IRF",
        default_value=1,
    ).tag(config=True)

    source_offset_n_edges = Integer(
        help="Number of edges for Source offset for PSF IRF",
        default_value=101,
    ).tag(config=True)

    def fov_offset_bins(self):
        """
        Creates bins for single/multiple FoV offset.
        """
        fov_offset = (
            np.linspace(
                self.fov_offset_min,
                self.fov_offset_max,
                self.fov_offset_n_edges,
            )
            * u.deg
        )
        return fov_offset

    def source_offset_bins(self):
        """
        Creates bins for source offset for generating PSF IRF.
        Using the same binning as in pyirf example.
        """

        source_offset = (
            np.linspace(
                self.source_offset_min,
                self.source_offset_max,
                self.source_offset_n_edges,
            )
            * u.deg
        )
        return source_offset
