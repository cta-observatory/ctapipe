import shutil

import joblib
import numpy as np
import tables
from astropy.table.operations import hstack, vstack
from ctapipe.core.tool import Tool
from ctapipe.core.traits import Bool, Path, create_class_enum_trait, flag
from ctapipe.io import TableLoader, write_table
from ctapipe.io.tableio import TelListToMaskTransform
from tqdm.auto import tqdm

from ..apply import EnergyRegressor, ParticleIdClassifier, Reconstructor
from ..sklearn import Classifier, Model, Regressor
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

    model_path = Path(
        default_value=None, allow_none=False, exists=True, directory_ok=False
    ).tag(config=True)

    stereo_combiner_type = create_class_enum_trait(
        base_class=StereoCombiner, default_value="StereoMeanCombiner"
    ).tag(config=True)

    aliases = {
        ("i", "input"): "Apply.input_url",
        ("m", "model"): "Apply.model_path",
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

        with open(self.model_path, "rb") as f:
            model, *_ = joblib.load(f)

        if isinstance(model, Regressor):
            self.reco = EnergyRegressor.read(
                self.model_path,
                self.loader.subarray,
                parent=self,
            )
            self.property = "energy"
        elif isinstance(model, Classifier):
            self.reco = ParticleIdClassifier.read(
                self.model_path,
                self.loader.subarray,
                parent=self,
            )
            self.property = "classification"
        else:
            raise TypeError(f"Unsupported model class {type(model)}")

        self.combine = StereoCombiner.from_name(
            self.stereo_combiner_type,
            combine_property=self.property,
            algorithm=self.reco.model.model_cls,
            parent=self,
        )

    def start(self):
        self.log.info("Applying model")
        prefix = self.reco.model.model_cls

        tables = []
        for tel_id, tel in tqdm(self.loader.subarray.tel.items()):
            if tel not in self.reco.model.models:
                self.log.warning(
                    "No model for telescope type %s, skipping tel %d",
                    tel,
                    tel_id,
                )
                continue

            table = self.loader.read_telescope_events([tel_id])
            table.remove_columns([c for c in table.colnames if c.startswith(prefix)])

            if len(table) == 0:
                self.log.warning("No events for telescope %d", tel_id)
                continue

            predictions = self.reco.predict(tel, table)
            table = hstack(
                [table, predictions], join_type="exact", metadata_conflicts="ignore"
            )

            write_table(
                table[["obs_id", "event_id", "tel_id"] + predictions.colnames],
                self.loader.input_url,
                f"/dl2/event/telescope/{self.property}/{prefix}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )
            tables.append(table)

        if len(tables) == 0:
            raise ValueError("No predictions made for any telescope")

        mono_predictions = vstack(tables)
        stereo_predictions = self.combine.predict(mono_predictions)
        trafo = TelListToMaskTransform(self.loader.subarray)
        for c in filter(
            lambda c: c.name.endswith("tel_ids"), stereo_predictions.columns.values()
        ):
            stereo_predictions[c.name] = np.array([trafo(r) for r in c])

        print(self.loader.input_url)
        write_table(
            stereo_predictions,
            self.loader.input_url,
            f"/dl2/event/subarray/{self.property}/{self.reco.model.model_cls}",
            mode="a",
            overwrite=self.overwrite,
        )

    def finish(self):
        self.h5file.close()


def main():
    Apply().run()


if __name__ == "__main__":
    main()
