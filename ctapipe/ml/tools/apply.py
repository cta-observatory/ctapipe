"""
Tool to apply machine learning models in bulk (as opposed to event by event).
"""
import shutil

import astropy.units as u
import numpy as np
import tables
from astropy.table.operations import hstack, vstack
from tqdm.auto import tqdm

from ctapipe.coordinates.disp import DispConverter
from ctapipe.core.tool import Tool
from ctapipe.core.traits import Bool, Path, create_class_enum_trait, flag
from ctapipe.io import EventSource, TableLoader, write_table
from ctapipe.io.tableio import TelListToMaskTransform

from ..sklearn import DispClassifier, DispRegressor, EnergyRegressor, ParticleIdClassifier
from ..stereo_combination import StereoCombiner


class ApplyModels(Tool):
    """Apply machine learning models to data.

    This tool predicts all events at once. To apply models in the
    regular event loop, set the appropriate options to ``ctapipe-process``.

    Models need to be trained with `~ctapipe.ml.tools.TrainEnergyRegressor`
    and `~ctapipe.ml.tools.TrainParticleIdClassifier`.
    """

    name = "ctapipe-ml-apply"
    description = __doc__

    overwrite = Bool(default_value=False).tag(config=True)

    input_url = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        exists=True,
    ).tag(config=True)

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)

    energy_regressor_path = Path(
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
    ).tag(config=True)

    particle_classifier_path = Path(
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
    ).tag(config=True)
    disp_regressor_path = Path(
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
    ).tag(config=True)
    sign_classifier_path = Path(
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
    ).tag(config=True)

    stereo_combiner_type = create_class_enum_trait(
        base_class=StereoCombiner,
        default_value="StereoMeanCombiner",
    ).tag(config=True)

    aliases = {
        ("i", "input"): "ApplyModels.input_url",
        "regressor": "ApplyModels.energy_regressor_path",
        "classifier": "ApplyModels.particle_classifier_path",
        "disp-regressor": "ApplyModels.disp_regressor_path",
        "sign-classifier": "ApplyModels.sign_classifier_path",
        ("o", "output"): "ApplyModels.output_path",
    }

    flags = {
        **flag(
            "overwrite",
            "ApplyModels.overwrite",
            "Overwrite tables in output file if it exists",
            "Don't overwrite tables in output file if it exists",
        ),
        "f": (
            {"ApplyModels": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
    }

    classes = [
        TableLoader,
        EnergyRegressor,
        ParticleIdClassifier,
        StereoCombiner,
    ]

    def setup(self):
        """
        Initialize components from config
        """
        self.log.info("Copying to output destination.")
        shutil.copy(self.input_url, self.output_path)

        self.h5file = tables.open_file(self.output_path, mode="r+")
        self.loader = TableLoader(
            parent=self,
            h5file=self.h5file,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2=True,
            load_simulated=True,
            load_instrument=True,
        )

        self.apply_regressor = self._setup_regressor()
        self.apply_classifier = self._setup_classifier()
        self.apply_geometry = self._setup_disp()

    def _setup_regressor(self):
        if self.energy_regressor_path is not None:
            self.energy_regressor = EnergyRegressor.read(
                self.energy_regressor_path,
                parent=self,
            )
            self.combine_energy = StereoCombiner.from_name(
                self.stereo_combiner_type,
                combine_property="energy",
                algorithm=self.energy_regressor.model_cls,
                log_target=True,
                parent=self,
            )
            return True
        return False

    def _setup_classifier(self):
        if self.particle_classifier_path is not None:
            self.classifier = ParticleIdClassifier.read(
                self.particle_classifier_path,
                parent=self,
            )
            self.combine_classification = StereoCombiner.from_name(
                self.stereo_combiner_type,
                combine_property="classification",
                algorithm=[self.classifier.model_cls],
                parent=self,
            )
            return True
        return False

    def _setup_disp(self):
        if (
            self.disp_regressor_path is not None
            and self.sign_classifier_path is not None
        ):
            self.disp_regressor = DispRegressor.read(
                self.disp_regressor_path, self.loader.subarray, parent=self
            )
            self.sign_classifier = DispClassifier.read(
                self.sign_classifier_path, self.loader.subarray, parent=self
            )
            self.disp_convert = DispConverter(parent=self)
            self.disp_combine = StereoCombiner.from_name(
                self.stereo_combiner_type,
                combine_property="geometry",
                algorithm=[
                    self.disp_regressor.model_cls,
                    self.sign_classifier.model_cls,
                ],
                parent=self,
            )
            return True
        return False

    def start(self):
        """Apply models to input tables"""
        if self.apply_regressor:
            self.log.info("Apply regressor.")
            mono_predictions = self._apply(self.energy_regressor, "energy")
            self._combine(self.combine_energy, mono_predictions)

        if self.apply_classifier:
            self.log.info("Apply classifier.")
            mono_predictions = self._apply(self.classifier, "classification")
            self._combine(self.combine_classification, mono_predictions)

        if self.apply_geometry:
            self.log.info("Apply disp reconstructors.")

            # Atm pointing information can only be accessed using EventSource
            # Pointing information will be added to HDF5 tables soon, see #1902
            for event in EventSource(self.input_url, max_events=1):
                self.disp_convert.pointing_altitude = event.pointing.array_altitude.to(
                    u.deg
                )
                self.disp_convert.pointing_azimuth = event.pointing.array_azimuth.to(
                    u.deg
                )

            mono_predictions = self._apply_disp()
            self._combine(self.disp_combine, mono_predictions)

    def _apply(self, reconstructor, parameter):
        prefix = reconstructor.model_cls

        tel_tables = []

        desc = f"Applying {reconstructor.__class__.__name__}"
        unit = "telescope"
        for tel_id, tel in tqdm(self.loader.subarray.tel.items(), desc=desc, unit=unit):
            if tel not in reconstructor._models:
                self.log.warning(
                    "No model in %s for telescope type %s, skipping tel %d",
                    reconstructor,
                    tel,
                    tel_id,
                )
                continue

            table = self.loader.read_telescope_events([tel_id])
            if len(table) == 0:
                self.log.warning("No events for telescope %d", tel_id)
                continue

            table.remove_columns([c for c in table.colnames if c.startswith(prefix)])

            predictions = reconstructor.predict_table(tel, table)
            table = hstack(
                [table, predictions],
                join_type="exact",
                metadata_conflicts="ignore",
            )
            write_table(
                table[["obs_id", "event_id", "tel_id"] + predictions.colnames],
                self.loader.input_url,
                f"/dl2/event/telescope/{parameter}/{prefix}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )
            tel_tables.append(table)

        if len(tel_tables) == 0:
            raise ValueError("No predictions made for any telescope")

        return vstack(tel_tables)

    def _apply_disp(self):
        # Different prefix? Two algorithms -> How to combine them?
        prefix = (
            self.disp_regressor.model.model_cls
            + "_"
            + self.sign_classifier.model.model_cls
        )

        tables = []

        for tel_id, tel in tqdm(self.loader.subarray.tel.items()):
            if tel not in self.disp_regressor.model.models:
                self.log.warning(
                    "No disp regressor model for telescope type %s, skipping tel %d",
                    tel,
                    tel_id,
                )
                continue
            if tel not in self.sign_classifier.model.models:
                self.log.warning(
                    "No sign classifier model for telescope type %s, skipping tel %d",
                    tel,
                    tel_id,
                )
                continue

            table = self.loader.read_telescope_events([tel_id])
            if len(table) == 0:
                self.log.warning("No events for telescope %d", tel_id)
                continue

            table.remove_columns(
                [
                    c
                    for c in table.colnames
                    if c.startswith(
                        (
                            prefix,
                            self.disp_regressor.model.model_cls + "_norm",
                            self.sign_classifier.model.model_cls + "_sign",
                        )
                    )
                ]
            )

            norm_predictions = self.disp_regressor.predict(tel, table)
            # If the same feature is generated for both models, it would cause an error
            table.remove_columns(
                [
                    c
                    for c in table.colnames
                    if np.logical_and(
                        c in self.disp_regressor.generate_features._feature_names,
                        c in self.sign_classifier.generate_features._feature_names,
                    )
                ]
            )
            sign_predictions = self.sign_classifier.predict(tel, table)
            table = hstack(
                [table, norm_predictions, sign_predictions],
                join_type="exact",
                metadata_conflicts="ignore",
            )

            altaz_predictions = self.disp_convert.predict(
                table,
                self.disp_regressor.model.model_cls,
                self.sign_classifier.model.model_cls,
            )
            table = hstack(
                [table, altaz_predictions],
                join_type="exact",
                metadata_conflicts="ignore",
            )

            # tables should follow same structure as ctapipe-process
            write_table(
                table[["obs_id", "event_id", "tel_id"] + altaz_predictions.colnames],
                self.loader.input_url,
                f"/dl2/event/telescope/geometry/{prefix}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )
            write_table(
                table[["obs_id", "event_id", "tel_id"] + norm_predictions.colnames],
                self.loader.input_url,
                f"/dl2/event/telescope/disp/{self.disp_regressor.model.model_cls}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )
            write_table(
                table[["obs_id", "event_id", "tel_id"] + sign_predictions.colnames],
                self.loader.input_url,
                f"/dl2/event/telescope/disp/{self.sign_classifier.model.model_cls}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )
            tables.append(table)

        if len(tables) == 0:
            raise ValueError("No predictions made for any telescope")

        return vstack(tables)

    def _combine(self, combiner, mono_predictions):
        stereo_predictions = combiner.predict_table(mono_predictions)

        trafo = TelListToMaskTransform(self.loader.subarray)
        for c in filter(
            lambda c: c.name.endswith("telescopes"),
            stereo_predictions.columns.values(),
        ):
            stereo_predictions[c.name] = np.array([trafo(r) for r in c])
            stereo_predictions[c.name].description = c.description

        if combiner.combine_property == "geometry":
            prefix = combiner.algorithm[0] + "_" + combiner.algorithm[1]
        else:
            prefix = combiner.algorithm[0]

        write_table(
            stereo_predictions,
            self.loader.input_url,
            f"/dl2/event/subarray/{combiner.combine_property}/{prefix}",
            mode="a",
            overwrite=self.overwrite,
        )

    def finish(self):
        """Close input file"""
        self.h5file.close()


def main():
    ApplyModels().run()


if __name__ == "__main__":
    main()
