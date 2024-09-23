"""
Component Wrappers around sklearn models
"""
import pathlib
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy

import astropy.units as u
import joblib
import numpy as np
from astropy.coordinates import AltAz
from astropy.table import QTable, Table, hstack
from astropy.utils.decorators import lazyproperty
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import all_estimators
from tables import open_file
from tqdm import tqdm
from traitlets import TraitError, observe

from ctapipe.exceptions import TooFewEvents

from ..containers import (
    ArrayEventContainer,
    DispContainer,
    ParticleClassificationContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
)
from ..coordinates import TelescopeFrame
from ..core import (
    Component,
    FeatureGenerator,
    Provenance,
    QualityQuery,
    ToolConfigurationError,
    traits,
)
from ..io import write_table
from .preprocessing import collect_features, table_to_X, telescope_to_horizontal
from .reconstructor import ReconstructionProperty, Reconstructor
from .stereo_combination import StereoCombiner
from .utils import add_defaults_and_meta

__all__ = [
    "SKLearnReconstructor",
    "SKLearnRegressionReconstructor",
    "SKLearnClassificationReconstructor",
    "EnergyRegressor",
    "ParticleClassifier",
    "DispReconstructor",
    "CrossValidator",
]


SUPPORTED_CLASSIFIERS = dict(all_estimators("classifier"))
SUPPORTED_REGRESSORS = dict(all_estimators("regressor"))
SUPPORTED_MODELS = {**SUPPORTED_CLASSIFIERS, **SUPPORTED_REGRESSORS}

_invalid_geometry = ReconstructedGeometryContainer(
    alt=u.Quantity(np.nan, unit=u.deg),
    az=u.Quantity(np.nan, unit=u.deg),
    is_valid=False,
)


class MLQualityQuery(QualityQuery):
    """Quality criteria for machine learning models with different defaults"""

    quality_criteria = traits.List(
        default_value=[
            ("> 50 phe", "hillas_intensity > 50"),
            ("Positive width", "hillas_width > 0"),
            ("> 3 pixels", "morphology_n_pixels > 3"),
            ("valid stereo reco", "HillasReconstructor_is_valid"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)


class SKLearnReconstructor(Reconstructor):
    """
    Base Class for a Machine Learning Based Reconstructor.

    Keeps a dictionary of sklearn models, the current tools are designed
    to train one model per telescope type.
    """

    #: Name of the target table column for training.
    target: str = ""

    #: Property predicted, overridden in subclass.
    property = None

    prefix = traits.Unicode(
        default_value=None,
        allow_none=True,
        help="Prefix for the output of this model. If None, ``model_cls`` is used.",
    ).tag(config=True)
    features = traits.List(
        traits.Unicode(), help="Features to use for this model."
    ).tag(config=True)
    model_config = traits.Dict({}, help="kwargs for the sklearn model.").tag(
        config=True
    )
    model_cls = traits.Enum(
        SUPPORTED_MODELS.keys(),
        default_value=None,
        allow_none=True,
        help="Which scikit-learn model to use.",
    ).tag(config=True)

    stereo_combiner_cls = traits.ComponentName(
        StereoCombiner,
        default_value="StereoMeanCombiner",
        help="Which stereo combination method to use.",
    ).tag(config=True)

    load_path = traits.Path(
        default_value=None,
        allow_none=True,
        help="If given, load serialized model from this path.",
    ).tag(config=True)

    def __init__(
        self, subarray=None, atmosphere_profile=None, models=None, n_jobs=None, **kwargs
    ):
        # Run the Component __init__ first to handle the configuration
        # and make `self.load_path` available
        Component.__init__(self, **kwargs)

        if self.load_path is None:
            if self.model_cls is None:
                raise TraitError(
                    "Must provide `model_cls` if not loading model from file"
                )

            if subarray is None:
                raise TypeError(
                    "__init__() missing 1 required positional argument: 'subarray'"
                )

            if self.prefix is None:
                # Default prefix is model_cls
                self.prefix = self.model_cls

            super().__init__(subarray, atmosphere_profile, **kwargs)
            self.feature_generator = FeatureGenerator(parent=self)
            self.quality_query = MLQualityQuery(parent=self)

            # to verify settings
            self._new_model()

            self._models = {} if models is None else models
            self.unit = None
            self.stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.prefix,
                property=self.property,
                parent=self,
            )
        else:
            loaded = self.read(self.load_path)
            if (
                subarray is not None
                and loaded.subarray.telescope_types != subarray.telescope_types
            ):
                self.log.warning(
                    "Supplied subarray has different telescopes than subarray loaded from file"
                )
            self.__dict__.update(loaded.__dict__)
            self.subarray = subarray

            if self.prefix is None:
                self.prefix = self.model_cls

    @abstractmethod
    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Event-wise prediction for the EventSource-Loop.

        Fills the event.dl2.<your-feature>[name] container.

        Parameters
        ----------
        event: ArrayEventContainer
        """

    @abstractmethod
    def predict_table(self, key, table: Table) -> Table:
        """
        Predict on a table of events.

        Parameters
        ----------
        key : Hashable
            Key of the model. Currently always a `~ctapipe.instrument.TelescopeDescription`
            as we train models per telescope type.
        table : `~astropy.table.Table`
            Table of features

        Returns
        -------
        table : `~astropy.table.Table`
            Table(s) with predictions, matches the corresponding
            container definition(s)
        """

    def write(self, path, overwrite=False):
        path = pathlib.Path(path)

        if path.exists() and not overwrite:
            raise OSError(f"Path {path} exists and overwrite=False")

        with path.open("wb") as f:
            Provenance().add_output_file(path, role="ml-models")
            joblib.dump(self, f, compress=True)

    @lazyproperty
    def instrument_table(self):
        return QTable(self.subarray.to_table("joined"))

    def _new_model(self):
        cfg = self.model_config
        if self.n_jobs:
            cfg["n_jobs"] = self.n_jobs
        return SUPPORTED_MODELS[self.model_cls](**cfg)

    def _table_to_y(self, table, mask=None):
        """
        Extract target values as numpy array from input table.
        """
        # make sure we use the unit that was used during training
        if self.unit is not None:
            return table[mask][self.target].quantity.to_value(self.unit)

        return np.array(table[self.target][mask])

    def fit(self, key, table):
        """
        Create and fit a new model for ``key`` using the data in ``table``.
        """
        self._models[key] = self._new_model()

        X, valid = table_to_X(table, self.features, self.log)
        self.unit = table[self.target].unit
        y = self._table_to_y(table, mask=valid)
        self._models[key].fit(X, y)

    @observe("n_jobs")
    def _set_n_jobs(self, n_jobs):
        """
        Update n_jobs of all associated models.
        """
        if hasattr(self, "_models"):
            for model in self._models.values():
                model.n_jobs = n_jobs.new


class SKLearnRegressionReconstructor(SKLearnReconstructor):
    """Base class for regression tasks."""

    model_cls = traits.Enum(
        SUPPORTED_REGRESSORS.keys(),
        default_value=None,
        allow_none=True,
        help="Which scikit-learn regression model to use.",
    ).tag(config=True)

    log_target = traits.Bool(
        default_value=False,
        help="If True, the model is trained to predict the natural logarithm.",
    ).tag(config=True)

    def _predict(self, key, table):
        if key not in self._models:
            raise KeyError(
                f"No model available for key {key},"
                f" available models: {self._models.keys()}"
            )
        X, valid = table_to_X(table, self.features, self.log)
        n_outputs = getattr(self._models[key], "n_outputs_", 1)

        if n_outputs > 1:
            shape = (len(table), n_outputs)
        else:
            shape = (len(table),)

        prediction = np.full(shape, np.nan)
        if np.any(valid):
            valid_predictions = self._models[key].predict(X)

            if self.log_target:
                prediction[valid] = np.exp(valid_predictions)
            else:
                prediction[valid] = valid_predictions

        if self.unit is not None:
            prediction = u.Quantity(prediction, self.unit, copy=False)

        return prediction, valid

    def _table_to_y(self, table, mask=None):
        y = super()._table_to_y(table, mask=mask)

        if self.log_target:
            if np.any(y <= 0):
                raise ValueError("y contains negative values, cannot apply log")

            return np.log(y)
        return y


class SKLearnClassificationReconstructor(SKLearnReconstructor):
    """Base class for classification tasks."""

    model_cls = traits.Enum(
        SUPPORTED_CLASSIFIERS.keys(),
        default_value=None,
        allow_none=True,
        help="Which scikit-learn classification model to use.",
    ).tag(config=True)

    invalid_class = traits.Integer(
        default_value=-1,
        help="The label value to fill in case no prediction could be made.",
    ).tag(config=True)

    positive_class = traits.Integer(
        default_value=1,
        help=(
            "The label value of the positive class in case of binary classification."
            " ``prediction`` values close to 1.0 mean the event"
            " belonged likely to this class."
        ),
    ).tag(config=True)

    def _predict(self, key, table):
        if key not in self._models:
            raise KeyError(
                f"No model available for key {key},"
                f" available models: {self._models.keys()}"
            )

        X, valid = table_to_X(table, self.features, self.log)
        n_outputs = getattr(self._models[key], "n_outputs_", 1)

        if n_outputs > 1:
            shape = (len(table), n_outputs)
        else:
            shape = (len(table),)

        prediction = np.full(shape, self.invalid_class, dtype=np.int8)
        if np.any(valid):
            prediction[valid] = self._models[key].predict(X)

        return prediction, valid

    def _predict_score(self, key, table):
        if key not in self._models:
            raise KeyError(
                f"No model available for key {key},"
                f" available models: {self._models.keys()}"
            )

        X, valid = table_to_X(table, self.features, self.log)

        n_classes = getattr(self._models[key], "n_classes_", 2)
        n_rows = len(table)
        shape = (n_rows, n_classes) if n_classes > 2 else (n_rows,)

        scores = np.full(shape, np.nan)

        if np.any(valid):
            prediction = self._models[key].predict_proba(X)[:]

            if n_classes > 2:
                scores[valid] = prediction
            else:
                # only return one score for the positive class
                scores[valid] = prediction[:, self._get_positive_index(key)]

        return scores, valid

    def _get_positive_index(self, key):
        return np.nonzero(self._models[key].classes_ == self.positive_class)[0][0]


class EnergyRegressor(SKLearnRegressionReconstructor):
    """
    Use a scikit-learn regression model per telescope type to predict primary energy.
    """

    target = "true_energy"
    property = ReconstructionProperty.ENERGY

    def __call__(self, event: ArrayEventContainer) -> None:
        for tel_id in event.trigger.tels_with_trigger:
            table = collect_features(event, tel_id, self.instrument_table)
            table = self.feature_generator(table, subarray=self.subarray)

            # get_table_mask returns a table with a single row
            passes_quality_checks = self.quality_query.get_table_mask(table)[0]

            if passes_quality_checks:
                prediction, valid = self._predict(
                    self.subarray.tel[tel_id],
                    table,
                )
                container = ReconstructedEnergyContainer(
                    energy=prediction[0],
                    is_valid=valid[0],
                )
            else:
                container = ReconstructedEnergyContainer(
                    energy=u.Quantity(np.nan, self.unit),
                    is_valid=False,
                )

            container.prefix = f"{self.prefix}_tel"
            event.dl2.tel[tel_id].energy[self.prefix] = container

        self.stereo_combiner(event)

    def predict_table(self, key, table: Table) -> dict[ReconstructionProperty, Table]:
        table = self.feature_generator(table, subarray=self.subarray)

        n_rows = len(table)
        energy = u.Quantity(np.full(n_rows, np.nan), self.unit, copy=False)
        is_valid = np.full(n_rows, False)

        valid = self.quality_query.get_table_mask(table)
        energy[valid], is_valid[valid] = self._predict(key, table[valid])

        result = Table(
            {
                f"{self.prefix}_tel_energy": energy,
                f"{self.prefix}_tel_is_valid": is_valid,
            }
        )
        add_defaults_and_meta(
            result,
            ReconstructedEnergyContainer,
            prefix=self.prefix,
            add_tel_prefix=True,
        )
        return {ReconstructionProperty.ENERGY: result}


class ParticleClassifier(SKLearnClassificationReconstructor):
    """Predict dl2 particle classification."""

    target = "true_shower_primary_id"

    positive_class = traits.Integer(
        default_value=0,
        help="Particle id (in simtel system) of the positive class. Default is 0 for gammas.",
    ).tag(config=True)

    property = ReconstructionProperty.PARTICLE_TYPE

    def __call__(self, event: ArrayEventContainer) -> None:
        for tel_id in event.trigger.tels_with_trigger:
            table = collect_features(event, tel_id, self.instrument_table)
            table = self.feature_generator(table, subarray=self.subarray)
            passes_quality_checks = self.quality_query.get_table_mask(table)[0]

            if passes_quality_checks:
                prediction, valid = self._predict_score(
                    self.subarray.tel[tel_id],
                    table,
                )

                container = ParticleClassificationContainer(
                    prediction=prediction[0],
                    is_valid=valid[0],
                )
            else:
                container = ParticleClassificationContainer(
                    prediction=np.nan, is_valid=False
                )

            container.prefix = f"{self.prefix}_tel"
            event.dl2.tel[tel_id].classification[self.prefix] = container

        self.stereo_combiner(event)

    def predict_table(self, key, table: Table) -> dict[ReconstructionProperty, Table]:
        table = self.feature_generator(table, subarray=self.subarray)

        n_rows = len(table)
        score = np.full(n_rows, np.nan)
        is_valid = np.full(n_rows, False)

        valid = self.quality_query.get_table_mask(table)
        score[valid], is_valid[valid] = self._predict_score(key, table[valid])

        result = Table(
            {
                f"{self.prefix}_tel_prediction": score,
                f"{self.prefix}_tel_is_valid": is_valid,
            }
        )
        add_defaults_and_meta(
            result,
            ParticleClassificationContainer,
            prefix=self.prefix,
            add_tel_prefix=True,
        )
        return {ReconstructionProperty.PARTICLE_TYPE: result}


class DispReconstructor(Reconstructor):
    """
    Predict absolute value and sign for disp origin reconstruction and
    convert to altitude and azimuth prediction for each telescope.
    """

    target = "true_disp"

    prefix = traits.Unicode(
        default_value="disp",
        allow_none=False,
        help="Prefix for the output of this model. If None, ``disp`` is used.",
    ).tag(config=True)

    features = traits.List(
        traits.Unicode(), help="Features to use for both models."
    ).tag(config=True)

    log_target = traits.Bool(
        default_value=False,
        help=(
            "If True, the norm(disp) model is trained to predict ln(norm(disp))"
            " and the output is"
            " ``prefix_parameter`` = ``sign_prediction`` * ``exp(norm_prediction)``."
        ),
    ).tag(config=True)

    norm_config = traits.Dict({}, help="kwargs for the sklearn regressor.").tag(
        config=True
    )

    norm_cls = traits.Enum(
        SUPPORTED_REGRESSORS.keys(),
        default_value=None,
        allow_none=True,
        help="Which scikit-learn regression model to use.",
    ).tag(config=True)

    sign_config = traits.Dict({}, help="kwargs for the sklearn classifier.").tag(
        config=True
    )

    sign_cls = traits.Enum(
        SUPPORTED_CLASSIFIERS.keys(),
        default_value=None,
        allow_none=True,
        help="Which scikit-learn classification model to use.",
    ).tag(config=True)

    stereo_combiner_cls = traits.ComponentName(
        StereoCombiner,
        default_value="StereoMeanCombiner",
        help="Which stereo combination method to use.",
    ).tag(config=True)

    load_path = traits.Path(
        default_value=None,
        allow_none=True,
        help="If given, load serialized model from this path.",
    ).tag(config=True)

    def __init__(self, subarray=None, atmosphere_profile=None, models=None, **kwargs):
        # Run the Component __init__ first to handle the configuration
        # and make `self.load_path` available
        Component.__init__(self, **kwargs)

        if self.load_path is None:
            if self.norm_cls is None or self.sign_cls is None:
                raise TraitError(
                    "Must provide `norm_cls` and `sign_cls` if not loading from file"
                )

            if subarray is None:
                raise TypeError(
                    "__init__() missing 1 required positional argument: 'subarray'"
                )

            super().__init__(subarray, atmosphere_profile, **kwargs)
            self.quality_query = MLQualityQuery(parent=self)
            self.feature_generator = FeatureGenerator(parent=self)

            # to verify settings
            self._new_models()
            self._models = {} if models is None else models
            self.unit = None
            self.stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.prefix,
                property=ReconstructionProperty.GEOMETRY,
                parent=self,
            )
        else:
            loaded = self.read(self.load_path)
            if (
                subarray is not None
                and loaded.subarray.telescope_types != subarray.telescope_types
            ):
                self.log.warning(
                    "Supplied subarray has different telescopes than subarray loaded from file"
                )
            self.__dict__.update(loaded.__dict__)
            self.subarray = subarray

    def _new_models(self):
        norm_cfg = self.norm_config
        sign_cfg = self.sign_config
        if self.n_jobs:
            norm_cfg["n_jobs"] = self.n_jobs
            sign_cfg["n_jobs"] = self.n_jobs
        norm_regressor = SUPPORTED_REGRESSORS[self.norm_cls](**norm_cfg)
        sign_classifier = SUPPORTED_CLASSIFIERS[self.sign_cls](**sign_cfg)
        return norm_regressor, sign_classifier

    def _table_to_y(self, table, mask=None):
        """
        Extract target values as numpy array from input table.
        """
        # make sure we use the unit that was used during training
        if self.unit is not None:
            norm = table[mask][self.target].quantity.to_value(self.unit)
        else:
            norm = np.array(table[self.target][mask])

        abs_norm = np.abs(norm)
        sign_norm = np.sign(norm)

        if self.log_target:
            abs_norm = np.log(abs_norm)

        return abs_norm, sign_norm

    def fit(self, key, table):
        """
        Create and fit new models for ``key`` using the data in ``table``.
        """
        self._models[key] = self._new_models()

        X, valid = table_to_X(table, self.features, self.log)
        self.unit = table[self.target].unit
        norm, sign = self._table_to_y(table, mask=valid)
        self._models[key][0].fit(X, norm)
        self._models[key][1].fit(X, sign)

    def write(self, path, overwrite=False):
        path = pathlib.Path(path)

        if path.exists() and not overwrite:
            raise OSError(f"Path {path} exists and overwrite=False")

        with path.open("wb") as f:
            Provenance().add_output_file(path, role="ml-models")
            joblib.dump(self, f, compress=True)

    @classmethod
    def read(cls, path, **kwargs):
        with open(path, "rb") as f:
            instance = joblib.load(f)

        for attr, value in kwargs.items():
            setattr(instance, attr, value)

        if not isinstance(instance, cls):
            raise TypeError(
                f"{path} did not contain an instance of {cls}, got {instance}"
            )

        # FIXME: we currently don't store metadata in the joblib / pickle files, see #2603
        Provenance().add_input_file(path, role="ml-models", add_meta=False)
        return instance

    @lazyproperty
    def instrument_table(self):
        return self.subarray.to_table("joined")

    def _predict(self, key, table):
        if key not in self._models:
            raise KeyError(
                f"No model available for key {key},"
                f" available models: {self._models.keys()}"
            )
        X, valid = table_to_X(table, self.features, self.log)
        prediction = np.full(len(table), np.nan)
        score = np.full(len(table), np.nan)

        if np.any(valid):
            valid_norms = self._models[key][0].predict(X)

            if self.log_target:
                prediction[valid] = np.exp(valid_norms)
            else:
                prediction[valid] = valid_norms

            sign_proba = self._models[key][1].predict_proba(X)[:, 0]
            # proba is [0 and 1] where 0 => very certain -1, 1 => very certain 1
            # and 0.5 means random guessing either. So we transform to a score
            # where 0 means "guessing" and 1 means "very certain"
            score[valid] = np.abs(2 * sign_proba - 1.0)
            prediction[valid] *= np.where(sign_proba >= 0.5, 1.0, -1.0)

        if self.unit is not None:
            prediction = u.Quantity(prediction, self.unit, copy=False)

        return prediction, score, valid

    def __call__(self, event: ArrayEventContainer) -> None:
        """
        Event-wise prediction for the EventSource-Loop.

        Fills the event.dl2.tel[tel_id].disp[prefix] container
        and event.dl2.tel[tel_id].geometry[prefix] container.

        Parameters
        ----------
        event: ArrayEventContainer
        """
        for tel_id in event.trigger.tels_with_trigger:
            table = collect_features(event, tel_id, self.instrument_table)
            table = self.feature_generator(table, subarray=self.subarray)

            passes_quality_checks = self.quality_query.get_table_mask(table)[0]

            if passes_quality_checks:
                disp, sign_score, valid = self._predict(
                    self.subarray.tel[tel_id], table
                )

                if valid:
                    disp_container = DispContainer(
                        parameter=disp[0],
                        sign_score=sign_score[0],
                    )

                    hillas = event.dl1.tel[tel_id].parameters.hillas
                    psi = hillas.psi.to_value(u.rad)

                    fov_lon = hillas.fov_lon + disp[0] * np.cos(psi)
                    fov_lat = hillas.fov_lat + disp[0] * np.sin(psi)
                    altaz = TelescopeFrame(
                        fov_lon=fov_lon,
                        fov_lat=fov_lat,
                        telescope_pointing=AltAz(
                            alt=event.pointing.tel[tel_id].altitude,
                            az=event.pointing.tel[tel_id].azimuth,
                        ),
                    ).transform_to(AltAz())

                    altaz_container = ReconstructedGeometryContainer(
                        alt=altaz.alt, az=altaz.az, is_valid=True
                    )

                else:
                    disp_container = DispContainer(
                        parameter=u.Quantity(np.nan, self.unit),
                    )
                    altaz_container = deepcopy(_invalid_geometry)
            else:
                disp_container = DispContainer(
                    parameter=u.Quantity(np.nan, self.unit),
                )
                altaz_container = deepcopy(_invalid_geometry)

            disp_container.prefix = f"{self.prefix}_tel"
            altaz_container.prefix = f"{self.prefix}_tel"
            event.dl2.tel[tel_id].disp[self.prefix] = disp_container
            event.dl2.tel[tel_id].geometry[self.prefix] = altaz_container

        self.stereo_combiner(event)

    def predict_table(self, key, table: Table) -> dict[ReconstructionProperty, Table]:
        """
        Predict on a table of events.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table of features

        Returns
        -------
        disp_table : `~astropy.table.Table`
            Table with disp predictions, matches the corresponding
            container definition
        altaz_table : `~astropy.table.Table`
            Table with resulting predictions of horizontal coordinates
        """
        table = self.feature_generator(table, subarray=self.subarray)

        n_rows = len(table)
        disp = u.Quantity(np.full(n_rows, np.nan), self.unit, copy=False)
        is_valid = np.full(n_rows, False)
        sign_score = np.full(n_rows, np.nan)

        valid = self.quality_query.get_table_mask(table)
        disp[valid], sign_score[valid], is_valid[valid] = self._predict(
            key, table[valid]
        )

        disp_result = Table(
            {
                f"{self.prefix}_tel_parameter": disp,
                f"{self.prefix}_tel_sign_score": sign_score,
            }
        )
        add_defaults_and_meta(
            disp_result,
            DispContainer,
            prefix=self.prefix,
            add_tel_prefix=True,
        )

        psi = table["hillas_psi"].quantity.to_value(u.rad)
        fov_lon = table["hillas_fov_lon"].quantity + disp * np.cos(psi)
        fov_lat = table["hillas_fov_lat"].quantity + disp * np.sin(psi)

        # prefer to use pointing interpolated to event
        if "telescope_pointing_altitude" in table.colnames:
            pointing_alt = table["telescope_pointing_altitude"]
            pointing_az = table["telescope_pointing_azimuth"]
        else:
            # fallback to fixed pointing of ob
            pointing_alt = table["subarray_pointing_lat"]
            pointing_az = table["subarray_pointing_lon"]

        alt, az = telescope_to_horizontal(
            lon=fov_lon,
            lat=fov_lat,
            pointing_alt=pointing_alt,
            pointing_az=pointing_az,
        )

        altaz_result = Table(
            {
                f"{self.prefix}_tel_alt": alt,
                f"{self.prefix}_tel_az": az,
                f"{self.prefix}_tel_is_valid": is_valid,
            }
        )
        add_defaults_and_meta(
            altaz_result,
            ReconstructedGeometryContainer,
            prefix=self.prefix,
            add_tel_prefix=True,
        )

        return {
            ReconstructionProperty.DISP: disp_result,
            ReconstructionProperty.GEOMETRY: altaz_result,
        }

    @observe("n_jobs")
    def _set_n_jobs(self, n_jobs):
        """
        Update n_jobs of all associated models.
        """
        if hasattr(self, "_models"):
            for disp, sign in self._models.values():
                disp.n_jobs = n_jobs.new
                sign.n_jobs = n_jobs.new


class CrossValidator(Component):
    """Class to train sklearn based reconstructors in a cross validation."""

    n_cross_validations = traits.Int(
        default_value=5, help="Number of cross validation iterations."
    ).tag(config=True)

    output_path = traits.Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
        help=(
            "Output path for the cross validation results."
            " This is a hdf5 file containing labels and predictions for"
            " the events used in the cross validation which can be used to"
            " create performance plots."
        ),
    ).tag(config=True)

    rng_seed = traits.Int(
        default_value=1337, help="Random seed for splitting the training data."
    ).tag(config=True)

    def __init__(self, model_component, overwrite=False, **kwargs):
        super().__init__(**kwargs)
        self.model_component = model_component
        self.rng = np.random.default_rng(self.rng_seed)

        if isinstance(self.model_component, SKLearnClassificationReconstructor):
            self.cross_validate = self._cross_validate_classification
            self.split_data = StratifiedKFold
        elif isinstance(self.model_component, SKLearnRegressionReconstructor):
            self.cross_validate = self._cross_validate_regressor
            self.split_data = KFold
        elif isinstance(self.model_component, DispReconstructor):
            self.cross_validate = self._cross_validate_disp
            self.split_data = StratifiedKFold
        else:
            raise KeyError(
                "Unsupported Model of type %s supplied", self.model_component
            )

        if self.output_path:
            if self.output_path.exists():
                if overwrite:
                    self.log.warning("Overwriting %s", self.output_path)
                else:
                    raise ToolConfigurationError(
                        f"Output path {self.output_path} exists, but overwrite=False"
                    )

            Provenance().add_output_file(self.output_path, role="ml-cross-validation")
            self.h5file = open_file(self.output_path, mode="w")

    def close(self):
        """Close the output hdf5 file, if ``self.output_path`` is given."""
        if self.output_path:
            self.h5file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __call__(self, telescope_type, table):
        """Perform cross validation for the given model."""
        if self.n_cross_validations == 0:
            return

        if len(table) <= self.n_cross_validations:
            raise TooFewEvents(f"Too few events for {telescope_type}.")

        self.log.info(
            "Starting cross-validation with %d folds for type %s.",
            self.n_cross_validations,
            telescope_type,
        )

        scores = defaultdict(list)
        kfold = self.split_data(
            n_splits=self.n_cross_validations,
            shuffle=True,
            # sklearn does not support numpy's new random API yet
            random_state=self.rng.integers(0, 2**31 - 1),
        )

        if isinstance(self.model_component, DispReconstructor):
            cv_it = kfold.split(table, np.sign(table[self.model_component.target]))
        else:
            cv_it = kfold.split(table, table[self.model_component.target])

        for fold, (train_indices, test_indices) in enumerate(
            tqdm(
                cv_it,
                total=self.n_cross_validations,
                desc=f"Cross Validation for {telescope_type}",
            ),
        ):
            train = table[train_indices]
            test = table[test_indices]

            cv_result, metrics = self.cross_validate(telescope_type, train, test)
            if self.output_path:
                results = Table(
                    data={
                        "cv_fold": np.full(len(cv_result), fold, dtype=np.uint8),
                        "tel_type": [str(telescope_type)] * len(cv_result),
                        "true_energy": test["true_energy"],
                        "true_impact_distance": test["true_impact_distance"],
                    },
                    descriptions={
                        "cv_fold": "Cross validation iteration",
                        "tel_type": "Telescope type",
                    },
                )
                results = hstack([results, cv_result], join_type="exact")
                write_table(
                    results,
                    self.h5file,
                    f"/cv_predictions/{telescope_type}",
                    append=True,
                )

            for metric, value in metrics.items():
                scores[metric].append(value)

        for metric, cv_values in scores.items():
            cv_values = np.array(cv_values)
            with np.printoptions(precision=4):
                self.log.info(
                    "Mean %s score from CV: %.4f ± %.4f",
                    metric,
                    cv_values.mean(),
                    cv_values.std(),
                )

    def _cross_validate_regressor(self, telescope_type, train, test):
        regressor = self.model_component
        regressor.fit(telescope_type, train)
        prediction, _ = regressor._predict(telescope_type, test)
        truth = test[regressor.target]
        r2 = r2_score(truth, prediction)
        result = Table(
            data={
                f"{regressor.prefix}_energy": prediction,
                "truth": truth,
            },
            descriptions={
                f"{regressor.prefix}_energy": "Predicted Energy",
                "truth": "Simulated Energy",
            },
        )
        return result, {"R^2": r2}

    def _cross_validate_classification(self, telescope_type, train, test):
        classifier = self.model_component
        classifier.fit(telescope_type, train)
        prediction, _ = classifier._predict_score(telescope_type, test)
        truth = np.where(
            test[classifier.target] == classifier.positive_class,
            1,
            0,
        )
        roc_auc = roc_auc_score(truth, prediction)
        result = Table(
            data={
                f"{classifier.prefix}_prediction": prediction,
                "truth": truth,
            },
            descriptions={
                f"{classifier.prefix}_prediction": "Predicted gammaness score",
                "truth": "Particle id (default is 1 for gammas)",
            },
        )
        return result, {"ROC AUC": roc_auc}

    def _cross_validate_disp(self, telescope_type, train, test):
        models = self.model_component
        models.fit(telescope_type, train)
        disp, sign_score, _ = models._predict(telescope_type, test)
        truth = test[models.target]
        r2 = r2_score(np.abs(truth), np.abs(disp))
        accuracy = accuracy_score(np.sign(truth), np.sign(disp))
        result = Table(
            data={
                f"{models.prefix}_parameter": disp,
                f"{models.prefix}_sign_score": sign_score,
                "truth": truth,
            },
            descriptions={
                f"{models.prefix}_parameter": "Predicted disp parameter",
                f"{models.prefix}_sign_score": "Score for how certain the disp sign classification was",
                "truth": "True disp parameter",
            },
        )
        return result, {"R^2": r2, "accuracy": accuracy}
