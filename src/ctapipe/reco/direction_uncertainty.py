import pathlib

import astropy.units as u
import joblib
import numpy as np
from astropy.coordinates import angular_separation
from astropy.table import Table

from ..core import FeatureGenerator, Provenance, QualityQuery, traits
from .preprocessing import table_to_X
from .reconstructor import Reconstructor
from .sklearn import SUPPORTED_REGRESSORS

__all__ = ["DirectionUncertaintyRegressor"]

from ..core.traits import TraitError, Unicode


class DirectionUncertaintyRegressor(Reconstructor):
    """
    Reconstructor for estimating the direction reconstruction uncertainty.
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
            raise TraitError("Must provide `model_cls` if not loading model from file")

        self.feature_generator = FeatureGenerator(parent=self)
        self.quality_query = QualityQuery(parent=self)
        # to verify settings
        self._new_model()

        self.unit = None

    def __call__(self, event):
        pass

    def predict_subarray_table(self, table):
        quality_valid = self.quality_query.get_table_mask(table)
        table = self.feature_generator(table[quality_valid], subarray=self.subarray)
        X, valid = table_to_X(table, self.features, self.log)
        n_rows = len(table)
        dir_uncert = np.full(n_rows, np.nan)
        dir_uncert[valid] = self._model.predict(X)
        dir_uncert = u.Quantity(dir_uncert, self.unit, copy=False)

        result = Table(
            {
                f"{self.reconstructor_prefix}_ang_distance_uncert": dir_uncert,
                f"{self.reconstructor_prefix}_is_valid": valid,
            }
        )
        return result

    def fit(self, table):
        """
        Create and fit a new model.
        """
        self._model = self._new_model()
        table = self.feature_generator(table, subarray=self.subarray)
        X, valid = table_to_X(table, self.features, self.log)
        dir_uncert = self._compute_angular_separation(table[valid])
        self.unit = dir_uncert.unit
        self._model.fit(X, dir_uncert.quantity.to_value(self.unit))

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
