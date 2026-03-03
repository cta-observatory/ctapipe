"""Module containing classes related to event loading and preprocessing"""

from astropy.coordinates import angular_separation

from ..coordinates import altaz_to_nominal
from ..core import (
    Component,
    FeatureGenerator,
    QualityQuery,
    ToolConfigurationError,
    traits,
)

__all__ = ["EventPreprocessor"]


from typing import Callable


class FeatureSetRegistry:
    """Registry for custom feature set configurations."""

    _registry = {}

    @classmethod
    def register(cls, name: str):
        """Register a feature set configuration.

        Examples
        --------
        >>> @FeatureSetRegistry.register("my_analysis")
        ... def my_config(preprocessor):
        ...     return {
        ...         "features_to_generate": [("custom", "col_a / col_b")],
        ...         "quality_criteria": [("cut", "custom > 0.5")],
        ...         "output_features": ["event_id", "custom"]
        ...     }
        """

        def decorator(func: Callable):
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str):
        """Get a registered configuration function."""
        return cls._registry.get(name)

    @classmethod
    def list_available(cls):
        """List all registered feature set names."""
        return list(cls._registry.keys())


@FeatureSetRegistry.register("dl2_irf")
def _dl2_irf_config(preprocessor):
    """Built-in configuration for DL2 IRF generation."""
    return {
        "features_to_generate": [
            ("reco_energy", f"{preprocessor.energy_reconstructor}_energy"),
            ("reco_alt", f"{preprocessor.geometry_reconstructor}_alt"),
            ("reco_az", f"{preprocessor.geometry_reconstructor}_az"),
            ("gh_score", f"{preprocessor.gammaness_reconstructor}_prediction"),
            ("theta", "angular_separation(reco_az, reco_alt, true_az, true_alt)"),
            (
                "reco_fov_coord",
                "altaz_to_nominal(reco_az, reco_alt, subarray_pointing_lon, subarray_pointing_lat)",
            ),
            (
                "reco_fov_lon",
                "reco_fov_coord[:,0]",
            ),  # note: GADF IRFs use the negative of this
            ("reco_fov_lat", "reco_fov_coord[:,1]"),
            (
                "true_fov_coord",
                "altaz_to_nominal(true_az, true_alt, subarray_pointing_lon, subarray_pointing_lat)",
            ),
            (
                "true_fov_lon",
                "true_fov_coord[:,0]",
            ),  # note: GADF IRFs use the negative of this
            ("true_fov_lat", "true_fov_coord[:,1]"),
            (
                "true_fov_offset",
                "angular_separation(true_fov_lon, true_fov_lat, 0*u.deg, 0*u.deg)",
            ),
            (
                "reco_fov_offset",
                "angular_separation(reco_fov_lon, reco_fov_lat, 0*u.deg, 0*u.deg)",
            ),
            (
                "multiplicity",
                f"np.count_nonzero({preprocessor.gammaness_reconstructor}_telescopes,axis=1)",
            ),
        ],
        "quality_criteria": [
            ("Valid geometry", f"{preprocessor.geometry_reconstructor}_is_valid"),
            ("valid energy", f"{preprocessor.energy_reconstructor}_is_valid"),
            ("valid gammaness", f"{preprocessor.gammaness_reconstructor}_is_valid"),
            ("sufficient multiplicity", "multiplicity >= 4"),
        ],
        "output_features": [
            "event_id",
            "obs_id",
            "reco_energy",
            "reco_alt",
            "reco_az",
            "gh_score",
            "true_energy",
            "true_alt",
            "true_az",
            "true_fov_offset",
            "reco_fov_offset",
            "theta",
            "reco_fov_lat",
            "true_fov_lat",
            "reco_fov_lon",
            "true_fov_lon",
            "multiplicity",
        ],
    }


class EventPreprocessor(Component):
    """
    Selects or generates features and filters tables of events.

    In normal use, one only has to specify the ``feature_set`` option, which
    will generate features supports standard use cases. For advanced usage, you
    can set ``feature_set=custom`` and pass in a configured
    `~ctapipe.core.FeatureGenerator` and set the ``features`` property of this
    class with the columns you to retain in the output table.

    In the `~ctapipe.core.FeatureGenerator` used internally, you have access to
    several additional functions useful for DL2 processing:

    - `~astropy.coordinates.angular_separation`
    - `~ctapipe.coordinates.altaz_to_nominal`
    """

    energy_reconstructor = traits.Unicode(
        default_value="RandomForestRegressor",
        help="Prefix of the reco `_energy` column",
    ).tag(config=True)

    geometry_reconstructor = traits.Unicode(
        default_value="HillasReconstructor",
        help="Prefix of the `_alt` and `_az` reco geometry columns",
    ).tag(config=True)

    gammaness_reconstructor = traits.Unicode(
        default_value="RandomForestClassifier",
        help="Prefix of the classifier `_prediction` column",
    ).tag(config=True)

    feature_set = traits.CaselessStrEnum(
        ["custom"] + FeatureSetRegistry.list_available(),
        default_value="custom",
        help=(
            "Set up the FeatureGenerator.features, output features, and quality criteria "
            "based on standard use cases."
            "Specify 'custom' if you want to set your own in your config file. If this is set to "
            "any value other than 'custom', the feature properties of the configuration "
            "file you pass in will be overridden."
        ),
    ).tag(config=True)

    features = traits.List(
        traits.Unicode(),
        help=(
            "Features (columns) to retain in the output.  "
            "These can include columns generated by the FeatureGenerator. "
            "If you set these, make sure feature_set=custom."
        ),
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        if self.feature_set == "custom":
            self.feature_generator = FeatureGenerator(parent=self)
            self.quality_query = QualityQuery(parent=self)
        else:  # use a pre-registered feature set
            feature_set = FeatureSetRegistry.get(self.feature_set)(self)
            self.feature_generator = FeatureGenerator(
                parent=self, features=feature_set["features_to_generate"]
            )
            self.quality_query = QualityQuery(
                parent=self, quality_criteria=feature_set["quality_criteria"]
            )
            self.features = feature_set["output_features"]
        # sanity checks:
        if len(self.features) == 0:
            raise ToolConfigurationError(
                "DL2EventPreprocessor has no output features configured."
                "You have set `feature_set=custom`, but did not provide the list "
                "of features in the configuration (DL2EventPreprocessor.features)."
            )

    def __call__(self, table):
        """Return new table with only the columns in features."""

        # generate new features, which includes renaming columns:
        generated = self.feature_generator(
            table,
            angular_separation=angular_separation,
            altaz_to_nominal=altaz_to_nominal,
        )

        # apply event selection on the resulting table

        selected_mask = self.quality_query.get_table_mask(generated)

        # return only the columns specified in `self.features`, and rows in
        # `selected_mask`
        return generated[self.features][selected_mask]
