"""
Define a parent IrfTool class to hold all the options
"""
import astropy.units as u
import numpy as np
from astropy.table import QTable
from pyirf.binning import create_bins_per_decade

from ..core import Component, QualityQuery, traits
from ..core.traits import Bool, Float, Integer, List, Unicode


class ToolConfig(Component):

    gamma_file = traits.Path(
        default_value=None, directory_ok=False, help="Gamma input filename and path"
    ).tag(config=True)
    gamma_sim_spectrum = traits.Unicode(
        default_value="CRAB_HEGRA",
        help="Name of the pyrif spectra used for the simulated gamma spectrum",
    ).tag(config=True)
    proton_file = traits.Path(
        default_value=None, directory_ok=False, help="Gamma input filename and path"
    ).tag(config=True)
    proton_sim_spectrum = traits.Unicode(
        default_value="IRFDOC_PROTON_SPECTRUM",
        help="Name of the pyrif spectra used for the simulated proton spectrum",
    ).tag(config=True)
    electron_file = traits.Path(
        default_value=None, directory_ok=False, help="Gamma input filename and path"
    ).tag(config=True)
    electron_sim_spectrum = traits.Unicode(
        default_value="IRFDOC_ELECTRON_SPECTRUM",
        help="Name of the pyrif spectra used for the simulated electron spectrum",
    ).tag(config=True)

    chunk_size = Integer(
        default_value=100000,
        allow_none=True,
        help="How many subarray events to load at once for making predictions.",
    ).tag(config=True)

    output_path = traits.Path(
        default_value="./IRF.fits.gz",
        allow_none=False,
        directory_ok=False,
        help="Output file",
    ).tag(config=True)

    overwrite = Bool(
        False,
        help="Overwrite the output file if it exists",
    ).tag(config=True)

    obs_time = Float(default_value=50.0, help="Observation time").tag(config=True)
    obs_time_unit = Unicode(
        default_value="hour",
        help="Unit used to specify observation time as an astropy unit string.",
    ).tag(config=True)

    alpha = Float(
        default_value=5.0, help="Ratio between size of on and off regions"
    ).tag(config=True)
    ON_radius = Float(default_value=1.0, help="Radius of ON region in degrees").tag(
        config=True
    )

    max_bg_radius = Float(
        default_value=5.0, help="Radius used to calculate background rate in degrees"
    ).tag(config=True)

    max_gh_cut_efficiency = Float(
        default_value=0.8, help="Maximum gamma purity requested"
    ).tag(config=True)
    gh_cut_efficiency_step = Float(
        default_value=0.1,
        help="Stepsize used for scanning after optimal gammaness cut",
    ).tag(config=True)
    initial_gh_cut_efficency = Float(
        default_value=0.4, help="Start value of gamma purity before optimisation"
    ).tag(config=True)


class EventPreProcessor(Component):
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

    preselect_criteria = List(
        default_value=[
            #            ("multiplicity 4", "np.count_nonzero(tels,axis=1) >= 4"),
            ("valid classifier", "valid_classer"),
            ("valid geom reco", "valid_geom"),
            ("valid energy reco", "valid_energy"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)

    rename_columns = List(
        help="List containing translation pairs of quality columns"
        "used for quality filters and their names as given in the input file used."
        "Ex: [('valid_geom','HillasReconstructor_is_valid')]",
        default_value=[
            ("valid_geom", "HillasReconstructor_is_valid"),
            ("valid_energy", "RandomForestRegressor_is_valid"),
            ("valid_classer", "RandomForestClassifier_is_valid"),
        ],
    )

    def _preselect_events(self, events):
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

        for new, old in self.rename_columns:
            rename_from.append(old)
            rename_to.append(new)

        keep_columns.extend(rename_from)
        events = QTable(events[keep_columns], copy=False)
        events.rename_columns(rename_from, rename_to)
        keep = QualityQuery(quality_criteria=self.preselect_criteria).get_table_mask(
            events
        )

        return events[keep]

    def _make_empty_table(self):
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
            "true_source_fov_offset",
            "reco_source_fov_offset",
            "weights",
        ]
        units = [
            None,
            None,
            u.TeV,
            u.deg,
            u.deg,
            u.TeV,
            u.deg,
            u.deg,
            None,
            u.deg,
            u.deg,
            u.deg,
            u.deg,
            None,
        ]

        return QTable(names=columns, units=units)


class ThetaSettings(Component):

    min_angle = Float(
        default_value=0.05, help="Smallest angular cut value allowed"
    ).tag(config=True)
    max_angle = Float(default_value=0.32, help="Largest angular cut value allowed").tag(
        config=True
    )
    min_counts = Integer(
        default_value=10,
        help="Minimum number of events in a bin to attempt to find a cut value",
    ).tag(config=True)
    fill_value = Float(
        default_value=0.32, help="Angular cut value used for bins with too few events"
    ).tag(config=True)


class DataBinning(Component):
    """
    Collects information on generating energy and angular bins for
    generating IRFs as per pyIRF requirements.

    Stolen from LSTChain
    """

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
        default_value=5,
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

    theta_min_angle = Float(
        default_value=0.05, help="Smallest angular cut value allowed"
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

    fov_offset_min = Float(
        help="Minimum value for FoV Offset bins in degrees",
        default_value=0.1,
    ).tag(config=True)

    fov_offset_max = Float(
        help="Maximum value for FoV offset bins in degrees",
        default_value=1.1,
    ).tag(config=True)

    fov_offset_n_edges = Integer(
        help="Number of edges for FoV offset bins",
        default_value=9,
    ).tag(config=True)

    bkg_fov_offset_min = Float(
        help="Minimum value for FoV offset bins for Background IRF",
        default_value=0,
    ).tag(config=True)

    bkg_fov_offset_max = Float(
        help="Maximum value for FoV offset bins for Background IRF",
        default_value=10,
    ).tag(config=True)

    bkg_fov_offset_n_edges = Integer(
        help="Number of edges for FoV offset bins for Background IRF",
        default_value=21,
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

    def true_energy_bins(self):
        """
        Creates bins per decade for true MC energy using pyirf function.
        The overflow binning added is not needed at the current stage.

        Examples
        --------
        It can be used as:

        >>> add_overflow_bins(***)[1:-1]
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
        The overflow binning added is not needed at the current stage.

        Examples
        --------
        It can be used as:

        >>> add_overflow_bins(***)[1:-1]
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

    def bkg_fov_offset_bins(self):
        """
        Creates bins for FoV offset for Background IRF,
        Using the same binning as in pyirf example.
        """
        background_offset = (
            np.linspace(
                self.bkg_fov_offset_min,
                self.bkg_fov_offset_max,
                self.bkg_fov_offset_n_edges,
            )
            * u.deg
        )
        return background_offset

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
