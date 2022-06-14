from astropy.table import Table
import tables
from ctapipe.core.tool import Tool
from ctapipe.core.traits import Bool, Path, flag
from ctapipe.io import TableLoader, write_table
from tqdm.auto import tqdm

from ..sklearn import Regressor
from ..apply import EnergyRegressor


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

    classes = [
        TableLoader,
        Regressor,
    ]

    def setup(self):
        """"""
        self.estimator = EnergyRegressor(
            parent=self,
            model=Regressor.load(self.model_path),
        )
        self.h5file = tables.open_file(self.input_url, mode="r+")
        self.loader = TableLoader(
            parent=self,
            h5file=self.h5file,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2_geometry=True,
            load_simulated=True,
        )

    def start(self):
        self.log.info("Applying model")

        for tel_id in tqdm(self.loader.subarray.tel):
            table = self.loader.read_telescope_events([tel_id])

            prediction, valid = self.estimator.predict(table)
            table = Table(
                {
                    "obs_id": table["obs_id"],
                    "event_id": table["event_id"],
                    "tel_id": table["tel_id"],
                    "reconstructed_energy_energy": prediction,
                    "reconstructed_energy_is_valid": valid,
                }
            )
            write_table(
                table,
                self.loader.input_url,
                f"/dl2/event/telescope/energy/{self.estimator.model.model_cls}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )

    def finish(self):
        self.h5file.close()


def main():
    ApplyEnergyRegressor().run()


if __name__ == "__main__":
    main()
