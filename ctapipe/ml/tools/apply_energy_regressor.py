from astropy.table.operations import hstack, vstack
import tables
from ctapipe.core.tool import Tool, ToolConfigurationError
from ctapipe.core.traits import Bool, Path, flag, create_class_enum_trait
from ctapipe.io import TableLoader, write_table
from tqdm.auto import tqdm
from ctapipe.io.tableio import TelListToMaskTransform
import numpy as np

from ..sklearn import Regressor
from ..apply import EnergyRegressor
from ..stereo_combination import StereoCombiner


class ApplyEnergyRegressor(Tool):

    overwrite = Bool(default_value=False).tag(config=True)

    input_url = Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
        exists=True,
    ).tag(config=True)

    model_path = Path(
        default_value=None, allow_none=False, exists=True, directory_ok=False
    ).tag(config=True)

    stereo_combiner_type = create_class_enum_trait(
        base_class=StereoCombiner, default_value="StereoMeanCombiner"
    ).tag(config=True)

    aliases = {
        ("i", "input"): "ApplyEnergyRegressor.input_url",
        ("m", "model"): "ApplyEnergyRegressor.model_path",
    }

    flags = {
        **flag(
            "overwrite",
            "ApplyEnergyRegressor.overwrite",
            "Overwrite tables in output file if it exists",
            "Don't overwrite tables in output file if it exists",
        ),
        "f": (
            {"ApplyEnergyRegressor": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
    }

    classes = [
        TableLoader,
        Regressor,
        StereoCombiner,
    ]

    def setup(self):
        """"""

        if self.input_url is None:
            raise ToolConfigurationError(
                "You must specify an input_url (--input / -i <URL>) !"
            )

        self.h5file = tables.open_file(self.input_url, mode="r+")
        self.loader = TableLoader(
            parent=self,
            h5file=self.h5file,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2=True,
            load_simulated=True,
            load_instrument=True,
        )
        self.regressor = EnergyRegressor.read(self.model_path, parent=self)
        self.combine = StereoCombiner.from_name(
            self.stereo_combiner_type,
            combine_property="energy",
            algorithm=self.regressor.model.model_cls,
            parent=self,
        )

    def start(self):
        self.log.info("Applying model")
        prefix = self.regressor.model.model_cls

        tables = []
        for tel_id, tel in tqdm(self.loader.subarray.tel.items()):
            if tel not in self.regressor.model.models:
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

            predictions = self.regressor.predict(tel, table)
            table = hstack(
                [table, predictions], join_type="exact", metadata_conflicts="ignore"
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

        mono_predictions = vstack(tables)
        stereo_predictions = self.combine.predict(mono_predictions)
        trafo = TelListToMaskTransform(self.loader.subarray)
        for c in filter(
            lambda c: c.name.endswith("tel_ids"), stereo_predictions.columns.values()
        ):
            stereo_predictions[c.name] = np.array([trafo(r) for r in c])

        write_table(
            stereo_predictions,
            self.loader.input_url,
            f"/dl2/event/subarray/energy/{self.regressor.model.model_cls}",
            mode="a",
            overwrite=self.overwrite,
        )

    def finish(self):
        self.h5file.close()


def main():
    ApplyEnergyRegressor().run()


if __name__ == "__main__":
    main()
