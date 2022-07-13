import shutil

import numpy as np
import tables
from astropy.table.operations import hstack, vstack
from tqdm.auto import tqdm

from ctapipe.core.tool import Tool
from ctapipe.core.traits import Bool, Path, create_class_enum_trait, flag
from ctapipe.io import TableLoader, write_table
from ctapipe.io.tableio import TelListToMaskTransform

from ..apply import EnergyRegressor, ParticleIdClassifier
from ..sklearn import Classifier, Regressor
from ..stereo_combination import StereoCombiner


class Apply(Tool):

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

    stereo_combiner_type = create_class_enum_trait(
        base_class=StereoCombiner,
        default_value="StereoMeanCombiner",
    ).tag(config=True)

    aliases = {
        ("i", "input"): "Apply.input_url",
        "regressor": "Apply.energy_regressor_path",
        "classifier": "Apply.particle_classifier_path",
        ("o", "output"): "Apply.output_path",
    }

    flags = {
        **flag(
            "overwrite",
            "Apply.overwrite",
            "Overwrite tables in output file if it exists",
            "Don't overwrite tables in output file if it exists",
        ),
        "f": (
            {"Apply": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
    }

    classes = [
        TableLoader,
        Regressor,
        Classifier,
        StereoCombiner,
    ]

    def setup(self):
        """"""
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

        if self.energy_regressor_path is not None:
            self.regressor = EnergyRegressor.read(
                self.energy_regressor_path,
                self.loader.subarray,
                parent=self,
            )
            self.regressor_combine = StereoCombiner.from_name(
                self.stereo_combiner_type,
                combine_property="energy",
                algorithm=self.regressor.model.model_cls,
                parent=self,
            )
            self.apply_regressor = True
        else:
            self.apply_regressor = False

        if self.particle_classifier_path is not None:
            self.classifier = ParticleIdClassifier.read(
                self.particle_classifier_path,
                self.loader.subarray,
                parent=self,
            )
            self.classifier_combine = StereoCombiner.from_name(
                self.stereo_combiner_type,
                combine_property="classification",
                algorithm=self.classifier.model.model_cls,
                parent=self,
            )
            self.apply_classifier = True
        else:
            self.apply_classifier = False

    @property
    def _has_classifier(self):
        return self.particle_classifier_path is not None

    @property
    def _has_regressor(self):
        return self.energy_regressor_path is not None

    def start(self):
        self.log.info("Applying model")
        if self._has_regressor:
            regressor_prefix = self.regressor.model.model_cls
        if self._has_classifier:
            classifier_prefix = self.classifier.model.model_cls

        tables = []
        for tel_id, tel in tqdm(self.loader.subarray.tel.items()):
            if self._has_regressor:
                if tel not in self.regressor.model.models:
                    self.log.warning(
                        "No regressor model for telescope type %s, skipping tel %d",
                        tel,
                        tel_id,
                    )
                    self.apply_regressor = False
                else:
                    self.apply_regressor = True
            if self._has_classifier:
                if tel not in self.classifier.model.models:
                    self.log.warning(
                        "No classifier model for telescope type %s, skipping tel %d",
                        tel,
                        tel_id,
                    )
                    self.apply_classifier = False
                else:
                    self.apply_classifier = True

            table = self.loader.read_telescope_events([tel_id])
            if self.apply_regressor:
                table.remove_columns(
                    [c for c in table.colnames if c.startswith(regressor_prefix)]
                )
            if self.apply_classifier:
                table.remove_columns(
                    [c for c in table.colnames if c.startswith(classifier_prefix)]
                )

            if len(table) == 0:
                self.log.warning("No events for telescope %d", tel_id)
                continue

            if self.apply_regressor:
                predictions = self.regressor.predict(tel, table)
                table = hstack(
                    [table, predictions],
                    join_type="exact",
                    metadata_conflicts="ignore",
                )
                write_table(
                    table[["obs_id", "event_id", "tel_id"] + predictions.colnames],
                    self.loader.input_url,
                    f"/dl2/event/telescope/energy/{regressor_prefix}/tel_{tel_id:03d}",
                    mode="a",
                    overwrite=self.overwrite,
                )
            if self.apply_classifier:
                predictions = self.classifier.predict(tel, table)
                table = hstack(
                    [table, predictions],
                    join_type="exact",
                    metadata_conflicts="ignore",
                )
                write_table(
                    table[["obs_id", "event_id", "tel_id"] + predictions.colnames],
                    self.loader.input_url,
                    f"/dl2/event/telescope/classification/{classifier_prefix}/tel_{tel_id:03d}",
                    mode="a",
                    overwrite=self.overwrite,
                )
            tables.append(table)

        if len(tables) == 0:
            raise ValueError("No predictions made for any telescope")

        mono_predictions = vstack(tables)
        if self.apply_regressor:
            stereo_predictions = self.regressor_combine.predict(mono_predictions)
            trafo = TelListToMaskTransform(self.loader.subarray)
            for c in filter(
                lambda c: c.name.endswith("tel_ids"),
                stereo_predictions.columns.values(),
            ):
                stereo_predictions[c.name] = np.array([trafo(r) for r in c])

            write_table(
                stereo_predictions,
                self.loader.input_url,
                f"/dl2/event/subarray/energy/{self.regressor.model.model_cls}",
                mode="a",
                overwrite=self.overwrite,
            )
        if self.apply_classifier:
            stereo_predictions = self.classifier_combine.predict(mono_predictions)
            trafo = TelListToMaskTransform(self.loader.subarray)
            for c in filter(
                lambda c: c.name.endswith("tel_ids"),
                stereo_predictions.columns.values(),
            ):
                stereo_predictions[c.name] = np.array([trafo(r) for r in c])

            write_table(
                stereo_predictions,
                self.loader.input_url,
                f"/dl2/event/subarray/classification/{self.classifier.model.model_cls}",
                mode="a",
                overwrite=self.overwrite,
            )

    def finish(self):
        self.h5file.close()


def main():
    Apply().run()


if __name__ == "__main__":
    main()
