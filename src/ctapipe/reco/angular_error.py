import pathlib

import joblib
from astropy.coordinates import angular_separation

from .preprocessing import table_to_X
from .reconstructor import Reconstructor
from ..core import traits, Provenance, FeatureGenerator
from .sklearn import SUPPORTED_REGRESSORS
__all__ = ['AngularErrorRegressor']

from ..core.traits import TraitError, Unicode


class AngularErrorRegressor(Reconstructor):
    """
    Reconstructor for estimating the angular reconstruction error.
    """
    features = traits.List(
        traits.Unicode(), help="Features to use for this model."
    ).tag(config=True)
    model_config = traits.Dict({}, help="kwargs for the sklearn model.").tag(
        config=True
    )
    model_cls = traits.Enum(
        SUPPORTED_REGRESSORS.keys(),
        default_value=None,
        allow_none=True,
        help="Which scikit-learn model to use.",
    ).tag(config=True)
    reconstructor_prefix = Unicode(
        default_value="HillasReconstructor",
        allow_none=False,
        help="Prefix of the reconstructor to use for the training.",
    ).tag(config=True)

    def __init__(self, subarray=None, models=None, n_jobs=None, **kwargs):
        # Run the Component __init__ first to handle the configuration
        # and make `self.load_path` available
        super().__init__(subarray, **kwargs)

        if self.model_cls is None:
            raise TraitError(
                "Must provide `model_cls` if not loading model from file"
            )

        self.feature_generator = FeatureGenerator(parent=self)

        # to verify settings
        self._new_model()

        self.unit = None

    def __call__(self, event):
        pass

    def predict_subarray_table(self, events):
        pass

    def fit(self, table):
        """
        Create and fit a new model.
        """
        self._model = self._new_model()
        table = self.feature_generator(table, subarray=self.subarray)
        X, valid = table_to_X(table, self.features, self.log)
        ang_error = self._compute_angular_separation(table[valid])
        self.unit = ang_error.unit
        self._model.fit(X, ang_error.quantity.to_value(self.unit))

    def _new_model(self):
        cfg = self.model_config
        return SUPPORTED_REGRESSORS[self.model_cls](**cfg)

    def _compute_angular_separation(self, table):
        return angular_separation(
            table[f"{self.reconstructor_prefix}_alt"],
            table[f"{self.reconstructor_prefix}_az"],
            table["true_alt"],
            table["true_az"],
        )

    def write(self, path, overwrite=False):
        path = pathlib.Path(path)

        if path.exists() and not overwrite:
            raise IOError(f"Path {path} exists and overwrite=False")

        with path.open("wb") as f:
            Provenance().add_output_file(path, role="ml-models")
            joblib.dump(self, f, compress=True)