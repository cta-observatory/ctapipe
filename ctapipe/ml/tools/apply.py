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

        self.apply_regressor = self._setup_regressor()
        self.apply_classifier = self._setup_classifier()

    def _setup_regressor(self):
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
            return True
        return False

    def _setup_classifier(self):
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
            return True
        return False

    def start(self):
        if self.apply_regressor:
            self.log.info("Apply regressor.")
            mono_predictions = self._apply_regressor()
            self._combine_regressor(mono_predictions)

        if self.apply_classifier:
            self.log.info("Apply classifier.")
            mono_predictions = self._apply_classifier()
            self._combine_classifier(mono_predictions)

    def _apply_regressor(self):
        prefix = self.regressor.model.model_cls

        tables = []

        for tel_id, tel in tqdm(self.loader.subarray.tel.items()):
            if tel not in self.regressor.model.models:
                self.log.warning(
                    "No regressor model for telescope type %s, skipping tel %d",
                    tel,
                    tel_id,
                )
                continue

            table = self.loader.read_telescope_events([tel_id])
            if len(table) == 0:
                self.log.warning("No events for telescope %d", tel_id)
                continue

            table.remove_columns([c for c in table.colnames if c.startswith(prefix)])

            predictions = self.regressor.predict(tel, table)
            table = hstack(
                [table, predictions],
                join_type="exact",
                metadata_conflicts="ignore",
            )
            write_table(
                table[["obs_id", "event_id", "tel_id"] + predictions.colnames],
                self.loader.input_url,
                f"/dl2/event/telescope/energy/{prefix}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )
            tables.append(table)

        if len(tables) == 0:
            raise ValueError("No predictions made for any telescope")

        return vstack(tables)

    def _combine_regressor(self, mono_predictions):
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

    def _apply_classifier(self):
        prefix = self.classifier.model.model_cls

        tables = []

        for tel_id, tel in tqdm(self.loader.subarray.tel.items()):
            if tel not in self.classifier.model.models:
                self.log.warning(
                    "No classifier model for telescope type %s, skipping tel %d",
                    tel,
                    tel_id,
                )
                continue

            table = self.loader.read_telescope_events([tel_id])
            if len(table) == 0:
                self.log.warning("No events for telescope %d", tel_id)
                continue

            table.remove_columns([c for c in table.colnames if c.startswith(prefix)])

            predictions = self.classifier.predict(tel, table)
            table = hstack(
                [table, predictions],
                join_type="exact",
                metadata_conflicts="ignore",
            )
            write_table(
                table[["obs_id", "event_id", "tel_id"] + predictions.colnames],
                self.loader.input_url,
                f"/dl2/event/telescope/classification/{prefix}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )
            tables.append(table)

        if len(tables) == 0:
            raise ValueError("No predictions made for any telescope")

        return vstack(tables)

    def _combine_classifier(self, mono_predictions):
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
