from astropy.table import Table
import tables
from ctapipe.core.tool import Tool
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
        allow_none=False,
        directory_ok=False,
        exists=True,
    ).tag(config=True)
    model_path = Path(
        default_value=None, allow_none=False, exists=True, directory_ok=False
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

    stereo_combiner_type = create_class_enum_trait(
        base_class=StereoCombiner, default_value="StereoMeanCombiner"
    )

    classes = [
        TableLoader,
        Regressor,
    ]

    def setup(self):
        """"""
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
        self.estimator = EnergyRegressor.read(
            self.model_path,
            self.loader.subarray,
            parent=self,
        )
        self.combine = StereoCombiner.from_name(
            self.stereo_combiner_type,
            combine_property='energy',
            algorithm=self.estimator.model.model_cls,
            parent=self,
        )

    def start(self):
        self.log.info("Applying model")

        for tel_id in tqdm(self.loader.subarray.tel):
            table = self.loader.read_telescope_events([tel_id])

            prediction, valid = self.estimator.predict(table)
            prefix = self.estimator.model.model_cls
            table = Table(
                {
                    "obs_id": table["obs_id"],
                    "event_id": table["event_id"],
                    "tel_id": table["tel_id"],
                    f"{prefix}_energy": prediction,
                    f"{prefix}_is_valid": valid,
                }
            )
            write_table(
                table,
                self.loader.input_url,
                f"/dl2/event/telescope/energy/{prefix}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )
        self.loader.load_instrument = False
        # TODO: Might need to turn this on if we want to use dl1 features as weights
        self.loader.load_dl1_parameters = False
        # TODO: Use chunks here once #1935 is merged and in the ml branch
        # TODO: This currently fails due to the bug described in #1938
        # TODO: The column selection is a quickfix to avoid issues with the check for valid rows
        # (time objects and all nan columns like uncertainty estimates)
        # We should either use only the valid flag from the reconstructor itself or
        # a proper quality query on the table
        mono_predictions = self.loader.read_telescope_events()
        stereo_predictions = self.combine.predict(mono_predictions)
        trafo = TelListToMaskTransform(self.loader.subarray)
        for k, c in filter(lambda c: c[0].endswith('tel_ids'), stereo_predictions.columns.items()):
            stereo_predictions[c.name] = np.array([trafo(r) for r in c])

        write_table(
            stereo_predictions,
            self.loader.input_url,
            f"/dl2/event/subarray/energy/{self.estimator.model.model_cls}",
            mode="a",
            overwrite=self.overwrite,
        )

    def finish(self):
        self.h5file.close()


def main():
    ApplyEnergyRegressor().run()


if __name__ == "__main__":
    main()
