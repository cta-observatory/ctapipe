from astropy.table import Table
from ctapipe.core.tool import Tool
from ctapipe.core.traits import Bool, Path, Unicode
from ctapipe.io import TableLoader, write_table
from ctapipe.ml.apply import EnergyRegressor

from ..sklearn import Regressor


class ApplyEnergyRegressor(Tool):

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)
    prediction_key = Unicode(default_value="pred_energy")
    overwrite = Bool(default_value=False).tag(config=True)

    aliases = {
        ("i", "input"): "TableLoader.input_url",
        ("o", "output"): "ApplyEnergyRegressor.output_path",
        ("m", "model"): "ApplyEnergyRegressor.model_path",
        ("f", "force"): "ApplyEnergyRegressor.overwrite",
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
        self.estimator = EnergyRegressor(parent=self)

    def start(self):
        self.log.info("Applying model")
        table = self.loader.read_telescope_events()

        prediction = Table({self.prediction_key: self.estimator.predict(table)})
        prediction[self.prediction_key].description = "Estimated energy"
        write_table(prediction, self.output_path, "/dl2/telescope/energy")

    def finish(self):
        self.loader.close()


if __name__ == "__main__":
    ApplyEnergyRegressor().run()
