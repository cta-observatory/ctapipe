from ctapipe.core.tool import Tool
from ctapipe.core.traits import Path, Unicode
from ctapipe.io import HDF5TableWriter, TableLoader

from ..sklearn import Regressor


class ApplyEnergyRegressor(Tool):

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)
    model_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)
    prediction = Unicode(default_value="pred_energy")

    aliases = {
        ("i", "input"): "TableLoader.input_url",
        ("o", "output"): "ApplyEnergyRegressor.output_path",
        ("m", "model"): "ApplyEnergyRegressor.model_path",
    }

    classes = [
        TableLoader,
        Regressor,
    ]

    def setup(self):
        """"""
        self.loader = TableLoader(
            parent=self,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2_geometry=True,
            load_simulated=True,
        )
        self.model = Regressor.load(self.model_path)

    def start(self):
        self.log.info("Applying model")
        table = self.loader.read_telescope_events()
        prediction = self.model.predict(table)
        table[self.prediction] = prediction
        table.write(self.output_path)

    def finish(self):
        self.loader.close()


if __name__ == "__main__":
    ApplyEnergyRegressor().run()
